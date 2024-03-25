use std::{
    collections::{BTreeMap, HashMap},
    iter,
};

use graphviz_rust::{
    dot_generator::edge,
    dot_structures::{
        Attribute, Edge, EdgeTy, Graph, GraphAttributes, Id, Node, NodeId, Stmt, Subgraph, Vertex,
    },
    printer::{DotPrinter, PrinterContext},
};
use itertools::Either;
use louds_rs::LoudsNodeNum;

use crate::token::{self, Token};

use super::{CompactFeatureModel, FeatureModel};
fn index_to_vertex(index: u64) -> Vertex {
    Vertex::N(NodeId(Id::Plain(format!("{index}")), None))
}

fn attribute(key: impl AsRef<str>, value: impl AsRef<str>) -> Attribute {
    let value = value.as_ref();
    let value_id = if value.contains('"') {
        Id::Escaped(value.to_string())
    } else {
        Id::Plain(value.to_string())
    };
    Attribute(Id::Plain(key.as_ref().to_string()), value_id)
}

fn index_pair_to_dot_edge(index1: u64, index2: u64) -> Edge {
    edge!(index_to_vertex(index1) => index_to_vertex(index2))
}

fn importance_to_color(importance: f32, activating: bool) -> String {
    let importance = ((1. - importance) * 255.0) as u8;
    if activating {
        format!("\"#ff{importance:02x}{importance:02x}\"")
    } else {
        format!("\"#{importance:02x}{importance:02x}ff\"")
    }
}

struct FeatureGraphNode {
    id: u64,
    importance: f32,
    end_node: bool,
    children: BTreeMap<Token, FeatureGraphNode>,
}

impl FeatureGraphNode {
    fn add_child(
        cur_children: &mut BTreeMap<Token, FeatureGraphNode>,
        token: Token,
        new_child: FeatureGraphNode,
    ) {
        match cur_children.get_mut(&token) {
            Some(self_node) => {
                self_node.merge(new_child);
            }
            None => {
                cur_children.insert(token, new_child);
            }
        }
    }

    fn merge(&mut self, other: Self) {
        self.importance = self.importance.max(other.importance);
        self.end_node = self.end_node || other.end_node;
        for (token, other_node) in other.children {
            Self::add_child(&mut self.children, token, other_node);
        }
    }

    fn from_model_louds_regular(
        model: &CompactFeatureModel,
        louds_num: LoudsNodeNum,
    ) -> FeatureGraphNode {
        let node = model.get_node(louds_num).expect("Invalid LOUDS number.");
        assert!(node.token.unpack().is_regular());
        let importance = node.importance;
        let end_node = node.end_node.is_some();
        let mut children = BTreeMap::new();
        for (token, child_louds_num) in model.children(louds_num) {
            match token.unpack() {
                crate::token::PatternToken::Regular(token) => {
                    let child = FeatureGraphNode::from_model_louds_regular(model, child_louds_num);
                    Self::add_child(&mut children, token, child);
                }
                crate::token::PatternToken::Ignore => {
                    for (token, child) in Self::from_model_louds_ignore(model, child_louds_num) {
                        Self::add_child(&mut children, token, child);
                    }
                }
            }
        }

        FeatureGraphNode {
            id: louds_num.0,
            importance,
            end_node,
            children,
        }
    }

    fn from_model_louds_ignore(
        model: &CompactFeatureModel,
        louds_num: LoudsNodeNum,
    ) -> impl Iterator<Item = (Token, FeatureGraphNode)> + '_ {
        let node = model.get_node(louds_num).expect("Invalid LOUDS number.");
        assert!(node.token.unpack().is_ignore());
        model
            .children(louds_num)
            .flat_map(|(token, child_louds_num)| match token.unpack() {
                crate::token::PatternToken::Regular(token) => {
                    let child = FeatureGraphNode::from_model_louds_regular(model, child_louds_num);
                    Either::Left(iter::once((token, child)))
                }
                crate::token::PatternToken::Ignore => Either::Right(
                    Self::from_model_louds_ignore(model, child_louds_num)
                        .collect::<Vec<_>>()
                        .into_iter(),
                ),
            })
    }
}

impl CompactFeatureModel {
    pub fn graphviz(&self, decode: impl Fn(Token) -> String) -> Graph {
        let roots: Vec<_> =
            FeatureGraphNode::from_model_louds_ignore(self, LoudsNodeNum(1)).collect();
        let mut layers: Vec<Vec<(Token, &FeatureGraphNode)>> =
            vec![roots.iter().map(|(token, node)| (*token, node)).collect()];
        loop {
            let last_layer = layers
                .last()
                .expect("There should be at least one layer")
                .as_slice();
            let new_layer: Vec<_> = last_layer
                .iter()
                .flat_map(|(_, node)| node.children.iter().map(|(token, node)| (*token, node)))
                .collect();
            if new_layer.is_empty() {
                break;
            }
            layers.push(new_layer);
        }
        let cluster_stmt_iter = layers
            .iter()
            .enumerate()
            .rev()
            .map(|(layer_index, nodes)| {
                let subgraph_index = layer_index + 1;
                let statements: Vec<Stmt> = nodes
                    .iter()
                    .map(|(token, node)| {
                        Stmt::Node(Node {
                            id: NodeId(Id::Plain(format!("{}", node.id)), None),
                            attributes: vec![
                                attribute("label", decode(*token)),
                                attribute(
                                    "fillcolor",
                                    importance_to_color(node.importance, subgraph_index == 1),
                                ),
                            ],
                        })
                    })
                    .collect();
                Subgraph {
                    id: Id::Plain(format!("cluster_{subgraph_index}")),
                    stmts: statements,
                }
            })
            .map(Stmt::Subgraph);
        let edge_stmt_iter = layers.iter().flatten().flat_map(|(_, node)| {
            node.children
                .values()
                .map(|child| Stmt::Edge(index_pair_to_dot_edge(child.id, node.id)))
        });
        let graph_attributes = Stmt::GAttribute(GraphAttributes::Graph(vec![
            attribute("nodesep", "0.2"),
            attribute("rankdir", "LR"),
            attribute("ranksep", "1.5"),
            attribute("splines", "spline"),
            attribute("pencolor", "white"),
            attribute("penwidth", "3"),
        ]));
        // Add more node attributes. E.g. shape, style and maybe penwidth. This removes them from the individual nodes and saves complexity/space.
        // If global attributes can be changed halfway through, use this for fontcolor.
        let node_attributes = Stmt::GAttribute(GraphAttributes::Node(vec![
            attribute("fixedsize", "true"),
            attribute("height", "0.75"),
            attribute("width", "2"),
            attribute("style", "\"filled,solid\""),
            attribute("shape", "box"),
            attribute("fontcolor", "black"),
            attribute("fontsize", "25"),
            attribute("penwidth", "7"),
        ]));
        let edge_attributes =
            Stmt::GAttribute(GraphAttributes::Edge(vec![attribute("penwidth", "3")]));
        let statements = [graph_attributes, node_attributes, edge_attributes]
            .into_iter()
            .chain(cluster_stmt_iter)
            .chain(edge_stmt_iter)
            .collect();
        Graph::DiGraph {
            id: Id::Anonymous("".to_string()),
            strict: false,
            stmts: statements,
        }
    }
}
