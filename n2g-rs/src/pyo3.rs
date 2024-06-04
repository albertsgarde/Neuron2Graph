use std::collections::BTreeMap;

use pyo3::prelude::*;

use crate::{
    feature_model::{CompactFeatureModel, FeatureModel, FeatureModelNode, Pattern},
    token::{CompactPatternToken, PatternToken, Token},
};

fn token_from_i32(token: i32) -> Token {
    Token(
        token
            .try_into()
            .unwrap_or_else(|token| panic!("Token negative: {token}")),
    )
}

#[pyclass(name = "Token")]
#[derive(Debug, Clone, Copy)]
pub struct PyToken {
    token: Token,
}

#[pymethods]
impl PyToken {
    #[staticmethod]
    fn from_i32(token: i32) -> Self {
        PyToken {
            token: token_from_i32(token),
        }
    }
}

impl From<PyToken> for Token {
    fn from(value: PyToken) -> Self {
        value.token
    }
}

#[pyclass(name = "PatternToken")]
#[derive(Debug, Clone, Copy)]
pub struct PyPatternToken {
    token: CompactPatternToken,
}

#[pymethods]
impl PyPatternToken {
    #[staticmethod]
    fn ignore() -> Self {
        PyPatternToken {
            token: PatternToken::Ignore.compact(),
        }
    }

    #[staticmethod]
    fn regular(token: i32) -> Self {
        PyPatternToken {
            token: PatternToken::Regular(token_from_i32(token)).compact(),
        }
    }
}

impl From<PyPatternToken> for CompactPatternToken {
    fn from(value: PyPatternToken) -> Self {
        value.token
    }
}

#[pyclass(name = "Pattern")]
#[derive(Clone)]
pub struct PyPattern {
    pattern: Pattern,
}

impl From<PyPattern> for Pattern {
    fn from(value: PyPattern) -> Self {
        value.pattern
    }
}

#[pymethods]
impl PyPattern {
    #[staticmethod]
    fn from_tokens(
        activating_token: i32,
        activating_importance: f32,
        context: Vec<(PyPatternToken, f32)>,
        activation: f32,
    ) -> Self {
        let activating_token = token_from_i32(activating_token);
        let context: Vec<(CompactPatternToken, _)> = context
            .into_iter()
            .map(|(token, importance)| (token.into(), importance))
            .collect();
        let pattern = Pattern::new(activating_token, activating_importance, context, activation);
        PyPattern { pattern }
    }
}

#[pyclass(name = "ModelNode")]
#[derive(Clone)]
pub struct PyModelNode {
    node: FeatureModelNode,
}

#[pymethods]
impl PyModelNode {
    #[staticmethod]
    pub fn from_children(
        children: Vec<(PyPatternToken, PyModelNode)>,
        importance: f32,
        end_node: Option<f32>,
    ) -> Self {
        let children: BTreeMap<CompactPatternToken, FeatureModelNode> = children
            .into_iter()
            .map(|(token, node)| (token.into(), node.into()))
            .collect();
        PyModelNode {
            node: FeatureModelNode::from_raw(end_node, children, importance),
        }
    }
}

impl From<PyModelNode> for FeatureModelNode {
    fn from(value: PyModelNode) -> Self {
        value.node
    }
}

#[pyclass(name = "FeatureModel")]
pub struct PyFeatureModel {
    model: CompactFeatureModel,
}

#[pymethods]
impl PyFeatureModel {
    #[staticmethod]
    pub fn from_patterns(patterns: Vec<PyPattern>) -> Self {
        let patterns: Vec<Pattern> = patterns.into_iter().map(|pattern| pattern.into()).collect();
        let model = (&FeatureModel::from_patterns(patterns)).into();
        PyFeatureModel { model }
    }

    #[staticmethod]
    pub fn from_nodes(nodes: Vec<(PyToken, PyModelNode)>) -> Self {
        let nodes: BTreeMap<Token, FeatureModelNode> = nodes
            .into_iter()
            .map(|(token, node)| (token.into(), node.into()))
            .collect();
        let model: CompactFeatureModel = FeatureModel::from_raw(nodes).into();
        PyFeatureModel { model }
    }

    pub fn __call__(&self, tokens: Vec<Vec<i32>>) -> Vec<Vec<f32>> {
        self.forward(tokens)
    }

    pub fn forward(&self, tokens: Vec<Vec<i32>>) -> Vec<Vec<f32>> {
        self.model
            .predict_activations(tokens.into_iter().map(|tokens| {
                tokens
                    .into_iter()
                    .map(token_from_i32)
                    .collect::<Vec<Token>>()
            }))
    }

    pub fn predict_activation(&self, tokens: Vec<i32>) -> Option<f32> {
        let (&activating, context) = tokens.split_first().unwrap();
        self.model.predict_activation(
            token_from_i32(activating),
            context.iter().map(|&token| token_from_i32(token)),
        )
    }
}

/// A Python module implemented in Rust.
#[pymodule]
fn n2g_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyToken>()?;
    m.add_class::<PyPatternToken>()?;
    m.add_class::<PyPattern>()?;
    m.add_class::<PyModelNode>()?;
    m.add_class::<PyFeatureModel>()?;
    Ok(())
}
