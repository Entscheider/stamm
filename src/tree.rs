//! Implements a very generic decision tree.
//! This means that the data for an interior node and for leafs can be anything.
//! The splitting criteria of the training set, the features used and the impurity computation can be implemented as required.
use serde::Serialize;
use serde::de::DeserializeOwned;


/// There a two cases for a binary tree: Using the left child or the right one.
/// So here we have a `Zero` or `One` to distinguish between these cases.
pub enum Binar {
    One,
    Zero,
}

/// A Node of a binary tree. It is an interior one or a leaf.
/// A leaf saves some data of type L, an interior node saves its children and some parameter of type T.
/// A leaf may wants to save something like the probability of being some class.
/// A interior node may wants to save some parameter which helps computing binary features.
#[derive(Serialize, Deserialize)]
enum Node<P, L> {
    Leaf(L),
    Interior {
        children: Box<(Node<P, L>, Node<P, L>)>,
        params: P,
    },
}


/// A trait for describing the behavior of a learned tree.
/// A Tree which follows this behavior can accept data of type `Data` and used parameter of type `Parameter` for interior nodes.
pub trait TreeFunction {
    type Data;
    type Param;

    /// Should compute the result of a binary feature for the `input` data using `param`.
    fn binarize(&self, param: &Self::Param, input: &Self::Data) -> Binar;

    /// Splits a list of input data `elements` using the parameter `param` for the binary feature.
    /// It returns a list where this binary feature is Zero and a list where this feature is One.
    fn split_set<'a>(
        &self,
        param: &Self::Param,
        elements: &[&'a Self::Data],
    ) -> (Vec<&'a Self::Data>, Vec<&'a Self::Data>) {
        let mut a = Vec::with_capacity(elements.len() / 2);
        let mut b = Vec::with_capacity(elements.len() / 2);
        for el in elements.iter() {
            match self.binarize(param, el) {
                Binar::Zero => a.push(*el),
                Binar::One => b.push(*el),
            }
        }
        (a, b)
    }
}

/// A trait for describing the behavior of a tree and the way it should be trained.
/// For training data of type `Truth` are used to describe the result this tree should have.
/// Data of type `LeafParam` are saved in leaves.
/// For generating feature where the best one should be chosen for training an interior node
/// an iterator of type `ParamIter` is used.
pub trait TreeLearnFunctions: TreeFunction {
    type Truth;
    type LeafParam;
    type ParamIter: Iterator<Item=Self::Param>;
    type PredictFunction: TreeFunction<Data=Self::Data, Param=Self::Param>;

/// Computes the impurity of two sets for selecting the best feature (where impurity has the lowest value).
/// The feature are described through `param`, the two sets are `set_l` and `set_r`.
/// `depth` is the depth of the current node this impurity should be calculated for.
    fn impurity(&self,
                param: &Self::Param,
                set_l: &[(&Self::Data, &Self::Truth)],
                set_r: &[(&Self::Data, &Self::Truth)],
                depth: usize)
                -> f64;

/// Generates some (finite!) parameters for a node.
/// The parameter which describes the best feature will be selected and saved in this node.
    fn param_set(&self) -> Self::ParamIter;

/// Computes the data which are saved in a leaf using the ground truth data `set` for this node.
    fn comp_leaf_data(&self, set: &[(&Self::Data, &Self::Truth)]) -> Self::LeafParam;

/// Like [`split_set`](trait.TreeFunction.html#method.split_set), but with ground truth data (List of `Self::Data` and `Self::Truth`).
    fn split_truth_set<'a>
        (&self,
         param: &Self::Param,
         elements: &[(&'a Self::Data, &'a Self::Truth)])
         -> (Vec<(&'a Self::Data, &'a Self::Truth)>, Vec<(&'a Self::Data, &'a Self::Truth)>) {
        let mut a = Vec::with_capacity(elements.len()/2);
        let mut b = Vec::with_capacity(elements.len()/2);
        for el in elements.iter() {
            let (x, truth) = *el;
            match self.binarize(param, x) {
                Binar::Zero => a.push((x, truth)),
                Binar::One => b.push((x, truth)),
            }
        }
        return (a, b);
    }

    /// Returns true if we want to build a leaf and do not want to divide `elements` in more parts.
    /// Returns false if we want to build a interior node and child nodes for dividing `elements` in some graduated parts.
    /// `depth` is the depth of the current node within the tree.
    #[allow(unused_variables)]
    fn early_stop(&self, depth: usize, elements: &[(&Self::Data, &Self::Truth)]) -> bool {
        false
    }

    /// Returns a TreeFunction which does not have to describe the training behavior anymore.
    /// As TreeLearnFunctions is also a TreeFunction self can be returned.
    /// However, there may be some reason to return another struct,
    /// e.g. if this TreeLearnFunctions contains a lot of data, which are not needed for using the learned tree.
    fn as_predict_learn_func(self) -> Self::PredictFunction;
}

/// Struct for learning a decision tree
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct TreeParameters {}

impl TreeParameters {
    pub fn new() -> TreeParameters {
        TreeParameters {}
    }
}

/// A decision tree for prediction all kinds of thing a decision tree can predict.
/// A leaf of this tree contains data of type `L`.
/// This tree uses a TreeFunction of Type `F`.
#[derive(Serialize, Deserialize)]
#[serde(bound(serialize = "F: Serialize,F::Param: Serialize, L: Serialize"))]
#[serde(bound(deserialize = "F: DeserializeOwned, F::Param: DeserializeOwned,L: DeserializeOwned"))]
pub struct DecisionTree<L, F>
where
    F: TreeFunction,
{
    root: Option<Node<F::Param, L>>, // root-Node
    functions: F, //  TreeFunction
    pub params: TreeParameters,
}

impl<L, F> DecisionTree<L, F>
where
    F: TreeFunction,
{
    /// Predict the output using the `input`.
    /// For doing that the input data will be tested using a binary feature of a node
    /// and depending on the result the feature of the left or right child node will be
    /// used for the next iteration. At the end there is a chosen leaf.
    /// Returns the data of the leaf the interior features have chosen.
    /// Returns None if the tree is invalid (e.g. no root-node).
    pub fn predict<'a>(&'a self, input: &F::Data) -> Option<&'a L> {
        if self.root.is_none() {
            return None;
        }
        let mut node: &Node<F::Param, L> = self.root.as_ref().unwrap();
        'l: loop {
            match *node {
                Node::Leaf(ref param) => return Some(param),
                Node::Interior {
                    ref children,
                    ref params,
                } => {
                    match self.functions.binarize(params, input) {
                        Binar::Zero => node = &(*children).0,
                        Binar::One => node = &(*children).1,
                    }
                }
            }
        }
    }
}

impl TreeParameters {
    /// Trains a decision tree using the TreeLearnFunctions `learn_func`
    /// and the ground truth data `train_set`.
    pub fn learn_tree<F>(
        self,
        learn_func: F,
        train_set: &[(&F::Data, &F::Truth)],
    ) -> DecisionTree<F::LeafParam,F::PredictFunction>
    where
        F: TreeLearnFunctions,
    {
        // For every subtree
        fn learn_tree_intern<F>(
            depth: usize,
            subset: &[(&F::Data, &F::Truth)],
            learn_func: &F,
        ) -> Node<F::Param, F::LeafParam>
        where
            F: TreeLearnFunctions,
        {
            use std::f64;
            assert!(subset.is_empty() == false);
            // Enough work? If so, make a leaf.
            if learn_func.early_stop(depth, subset) {
                return Node::Leaf(learn_func.comp_leaf_data(subset));
            }
            // Look for every parameter which describes a feature
            let parameters = learn_func.param_set();
            // Look for the best parameter/feature using impurity
            let mut best_impurity = f64::INFINITY;
            let mut best_param: Option<F::Param> = None;
            let mut left: Vec<(&F::Data, &F::Truth)> = vec![];
            let mut right: Vec<(&F::Data, &F::Truth)> = vec![];
            for param in parameters {
                let (left_, right_) = learn_func.split_truth_set(&param, subset);

                // If there is some set empty, the partition is not good
                if left_.is_empty() {
                    continue;
                }
                if right_.is_empty() {
                    continue;
                }

                // Impurity lesser than all before?
                let impurity = learn_func.impurity(&param, &left_[..], &right_[..], depth);
                assert!(false == impurity.is_nan());
                if impurity < best_impurity {
                    best_impurity = impurity;
                    best_param = Some(param);
                    left = left_;
                    right = right_;
                }
            }
            if let Some(best) = best_param {
                // Make a node using the best parameter
                Node::Interior {
                    params: best,
                    children: Box::new((
                        learn_tree_intern(depth + 1, &left[..], learn_func),
                        learn_tree_intern(depth + 1, &right[..], learn_func),
                    )),
                }
            } else {
                // No parameter partitions the input into two strict subsets. => Make a leaf
                Node::Leaf(learn_func.comp_leaf_data(subset))
            }
        }
        DecisionTree {
            root: Some(learn_tree_intern(0, train_set, &learn_func)),
            functions: learn_func.as_predict_learn_func(),
            params: self,
        }
    }
}
