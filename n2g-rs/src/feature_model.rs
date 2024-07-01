use std::{collections::BTreeMap, iter};

use louds_rs::{Louds, LoudsNodeNum};

use crate::token::{CompactPatternToken, Token};
use serde::{Deserialize, Serialize};

mod graphviz;

#[derive(Clone)]
pub struct Pattern {
    activating: (Token, f32),
    tokens: Vec<(CompactPatternToken, f32)>,
    activation: f32,
}

impl Pattern {
    pub fn new(
        activating_token: Token,
        activating_importance: f32,
        tokens: Vec<(CompactPatternToken, f32)>,
        activation: f32,
    ) -> Self {
        Pattern {
            activating: (activating_token, activating_importance),
            tokens,
            activation,
        }
    }
}

#[derive(Clone)]
pub struct FeatureModelNode {
    end_node: Option<f32>,
    children: BTreeMap<CompactPatternToken, FeatureModelNode>,
    importance: f32,
}

impl FeatureModelNode {
    pub fn new_token(importance: f32) -> Self {
        FeatureModelNode {
            end_node: None,
            children: BTreeMap::new(),
            importance,
        }
    }

    pub fn from_raw(
        end_node: Option<f32>,
        children: BTreeMap<CompactPatternToken, FeatureModelNode>,
        importance: f32,
    ) -> Self {
        FeatureModelNode {
            end_node,
            children,
            importance,
        }
    }
}

pub struct FeatureModel {
    children: BTreeMap<Token, FeatureModelNode>,
}

impl FeatureModel {
    pub fn new() -> FeatureModel {
        FeatureModel {
            children: BTreeMap::new(),
        }
    }

    pub fn from_raw(nodes: BTreeMap<Token, FeatureModelNode>) -> FeatureModel {
        FeatureModel { children: nodes }
    }

    pub fn from_patterns(patterns: impl IntoIterator<Item = Pattern>) -> FeatureModel {
        let mut model = FeatureModel::new();
        for pattern in patterns {
            model.add_pattern(&pattern);
        }
        model
    }

    pub fn add_pattern(&mut self, pattern: &Pattern) {
        let (activating, activating_importance) = pattern.activating;
        let mut cur_node = self
            .children
            .entry(activating)
            .or_insert_with(|| FeatureModelNode::new_token(activating_importance));
        cur_node.importance = cur_node.importance.max(activating_importance);
        for &(token, importance) in pattern.tokens.iter() {
            cur_node = cur_node
                .children
                .entry(token)
                .or_insert_with(|| FeatureModelNode::new_token(importance));
            cur_node.importance = cur_node.importance.max(importance);
        }
        cur_node.end_node = Some(pattern.activation);
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct CompactFeatureModelNode {
    token: CompactPatternToken,
    end_node: Option<f32>,
    importance: f32,
}

const DUMMY_NODE: CompactFeatureModelNode = CompactFeatureModelNode {
    token: CompactPatternToken::new_ignore(),
    end_node: None,
    importance: f32::NAN,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompactFeatureModel {
    louds: Louds,
    tokens: Vec<CompactFeatureModelNode>,
}

impl CompactFeatureModel {
    fn activating_nodes(
        &self,
    ) -> impl Iterator<Item = (LoudsNodeNum, CompactFeatureModelNode)> + '_ {
        let root_louds = LoudsNodeNum(1);
        self.louds
            .parent_to_children_nodes(root_louds)
            .map(|louds| {
                let node = self
                    .get_node(louds)
                    .expect("Children should always have nodes.");
                (louds, node)
            })
    }

    fn children(
        &self,
        index: LoudsNodeNum,
    ) -> impl Iterator<Item = (CompactPatternToken, LoudsNodeNum)> + '_ {
        self.louds
            .parent_to_children_nodes(index)
            .map(move |louds| {
                let node = self
                    .get_node(louds)
                    .expect("Children should always have nodes.");
                (node.token, louds)
            })
    }

    fn nodes(&self) -> impl Iterator<Item = (LoudsNodeNum, CompactFeatureModelNode)> + '_ {
        self.tokens
            .iter()
            .copied()
            .enumerate()
            .skip(2)
            .map(|(i, node)| {
                let louds = LoudsNodeNum(i.try_into().expect("Should never fail."));
                (louds, node)
            })
    }

    pub fn num_nodes(&self) -> usize {
        assert!(self.tokens.len() >= 2, "Model must have at least 2 nodes.");
        self.tokens.len() - 2
    }

    fn get_node(&self, node_num: LoudsNodeNum) -> Option<CompactFeatureModelNode> {
        assert_ne!(
            node_num,
            LoudsNodeNum(0),
            "LOUDS node number 0 is not valid."
        );
        assert_ne!(
            node_num,
            LoudsNodeNum(1),
            "Root node is virtual and has no data."
        );
        self.tokens.get(node_num.0 as usize).copied()
    }

    /// Predicts the activation for the given token given context.
    /// Context should be given in reverse order.
    pub fn predict_activation(
        &self,
        activating_token: Token,
        context: impl IntoIterator<Item = Token>,
    ) -> Option<f32> {
        let activating_token = CompactPatternToken::new_token(activating_token);
        let mut cur_node = self
            .activating_nodes()
            .find(|(_, node)| node.token == activating_token)?;
        for context_token in context.into_iter() {
            cur_node = if let Some(next_node) = self
                .louds
                .parent_to_children_nodes(cur_node.0)
                .map(|child_node| {
                    (
                        child_node,
                        self.get_node(child_node).unwrap_or_else(|| {
                            panic!(
                        "All children must exist. Child: {child_node:?}  Parent: {cur_node:?}"
                    )
                        }),
                    )
                })
                .find(|(_, child_token)| {
                    // Ignore tokens should be the last children, so this should be fine.
                    child_token.token.matches(context_token)
                }) {
                next_node
            } else {
                break;
            };
        }
        let (_, end_node) = cur_node;
        end_node.end_node
    }

    /// Predicts the activations for the given tokens.
    /// Each element of the outer output vector corresponds to one input sequence.
    /// For each input sequence, the corresponding vector is the predicted activation on each token given the preceding tokens as context.
    pub fn predict_activations<I1, I2>(&self, tokens: I1) -> Vec<Vec<f32>>
    where
        I2: AsRef<[Token]>,
        I1: IntoIterator<Item = I2>,
    {
        tokens
            .into_iter()
            .map(|tokens| {
                let tokens: &[Token] = tokens.as_ref();
                (1..=tokens.len())
                    .map(|i| {
                        let slice = &tokens[..i];
                        let (&activating, context) =
                            slice.split_last().expect("Slice cannot be empty.");
                        self.predict_activation(activating, context.iter().rev().copied())
                            .unwrap_or(0.)
                    })
                    .collect()
            })
            .collect()
    }

    pub fn tokens(&self) -> impl Iterator<Item = (Token, bool)> + '_ {
        let num_activating = self.activating_nodes().count();
        self.tokens
            .iter()
            .enumerate()
            .skip(2)
            .map(move |(i, node)| (node.token, i >= 2 + num_activating))
            .filter_map(|(token, is_activating)| {
                token.regular().map(|token| (token, is_activating))
            })
    }

    pub fn is_trie_equal(&self, other: &CompactFeatureModel) -> bool {
        self.nodes().zip(other.nodes()).all(
            |((self_louds_num, self_node), (other_louds_num, other_node))| {
                self_louds_num == other_louds_num
                    && self_node.end_node.is_some() == other_node.end_node.is_some()
                    && self_node.importance == other_node.importance
                    && self_node.token == other_node.token
            },
        )
    }
}

impl From<&FeatureModel> for CompactFeatureModel {
    fn from(trie: &FeatureModel) -> Self {
        fn lbs_part(num_children: usize) -> impl Iterator<Item = bool> {
            iter::repeat(true)
                .take(num_children)
                .chain(iter::once(false))
        }
        let mut bit_string: Vec<bool> = lbs_part(1).chain(lbs_part(trie.children.len())).collect();
        let mut tokens: Vec<CompactFeatureModelNode> = vec![DUMMY_NODE, DUMMY_NODE];
        let mut stack1: Vec<(CompactPatternToken, &FeatureModelNode)> = trie
            .children
            .iter()
            .map(|(token, node)| (CompactPatternToken::new_token(*token), node))
            .collect();
        let mut stack2: Vec<(CompactPatternToken, &FeatureModelNode)> = Vec::new();
        let mut cur_layer = &mut stack1;
        let mut next_layer = &mut stack2;
        while !cur_layer.is_empty() {
            for (token, node) in cur_layer.drain(..) {
                bit_string.extend(lbs_part(node.children.len()));
                next_layer.extend(node.children.iter().map(|(token, node)| (*token, node)));
                tokens.push(CompactFeatureModelNode {
                    token,
                    end_node: node.end_node,
                    importance: node.importance,
                });
            }
            std::mem::swap(&mut cur_layer, &mut next_layer);
        }
        let num_nodes = tokens.len() - 1;
        debug_assert_eq!(bit_string.iter().filter(|&&b| b).count(), num_nodes);
        debug_assert_eq!(bit_string.len(), 2 * num_nodes + 1);
        Self {
            louds: Louds::from(bit_string.as_slice()),
            tokens,
        }
    }
}

impl From<FeatureModel> for CompactFeatureModel {
    fn from(trie: FeatureModel) -> Self {
        Self::from(&trie)
    }
}

#[cfg(test)]
mod test {

    use crate::token::PatternToken;

    use super::*;

    #[test]
    fn empty_trie_compact() {
        let model = FeatureModel::new();
        let compact_model = CompactFeatureModel::from(&model);
        assert_eq!(compact_model.predict_activation(Token(432), []), None);
    }

    fn tiny_trie_model() -> FeatureModel {
        let patterns = [
            Pattern {
                activating: (Token(41), 0.8),
                tokens: vec![
                    (PatternToken::Ignore.compact(), 0.0),
                    (PatternToken::Regular(Token(52)).compact(), 0.5),
                ],
                activation: 0.7,
            },
            Pattern {
                activating: (Token(41), 1.0),
                tokens: vec![(PatternToken::Regular(Token(52)).compact(), 0.9)],
                activation: 0.5,
            },
            Pattern {
                activating: (Token(43), 1.0),
                tokens: vec![(PatternToken::Regular(Token(463)).compact(), 0.9)],
                activation: 0.4,
            },
        ];

        FeatureModel::from_patterns(patterns)
    }

    #[test]
    fn tiny_trie_compact() {
        let model = tiny_trie_model();
        let compact_model = CompactFeatureModel::from(&model);
        assert_eq!(compact_model.predict_activation(Token(41), []), None);
        assert_eq!(
            compact_model.predict_activation(Token(41), [Token(52)]),
            Some(0.5)
        );
        assert_eq!(
            compact_model.predict_activation(Token(41), [Token(52), Token(52)]),
            Some(0.5)
        );
        assert_eq!(
            compact_model.predict_activation(Token(41), [Token(564), Token(52)]),
            Some(0.7)
        );
        assert_eq!(
            compact_model.predict_activation(Token(41), [Token(564)]),
            None
        );
        assert_eq!(
            compact_model.predict_activation(Token(43), [Token(463), Token(521)]),
            Some(0.4)
        );
    }
}
