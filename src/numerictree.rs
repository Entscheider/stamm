//! NumericTree implements a decision tree for numeric computation using `stamm::tree`.

use super::tree::*;
use std::vec::IntoIter;
use rand;



/// There a two labels: in class or not in class
#[derive(Debug, Copy, Clone)]
pub enum NumericTruth {
    InClass,
    NotInClass,
}

impl NumericTruth {
    fn in_class(self) -> bool {
        match self {
            NumericTruth::InClass => true,
            _ => false,
        }
    }
}

/// A interior node saves a threshold for computing the feature.
#[derive(Serialize, Deserialize)]
pub struct NumericNodeParam {
    threshold: f64,
}

/// Every leaf saves the probability of being InClass
#[derive(Serialize, Deserialize)]
pub struct NumericLeafParam {
    pub probability: f64,
}


/// Impurity computation of a set (not two sets)
pub trait ImpurityFunc {
    fn impurity(&self, param: &[(&f64, &NumericTruth)]) -> f64;
}

/// Impurity using entropy
#[derive(Serialize, Deserialize)]
pub struct EntropyImpurity;
impl ImpurityFunc for EntropyImpurity {
    fn impurity(&self, set: &[(&f64, &NumericTruth)]) -> f64 {
        let abs = set.iter().fold(0, |sum, &(_, t)| match *t {
            NumericTruth::InClass => sum + 1,
            _ => sum,
        });
        let rel = (abs as f64) / (set.len() as f64);
        macro_rules! ln {
        ($x: expr) => {if $x == 0f64 {0f64} else {$x.ln()}}
    }
        return -(rel * ln!(rel) + (1f64 - rel) * ln!(1f64 - rel));
    }
}

/// Implements the feature of a numeric tree (meaning less or greater than a threshold).
#[derive(Serialize, Deserialize)]
pub struct NumericTreeFunc<F>
where
    F: ImpurityFunc,
{
    /// the minimum value expected  (for training)
    pub min_nr: f64,
    /// the maximum value expected (for training)
    pub max_nr: f64,
    /// number of thresholds generated for every interior node
    pub nr_feat_per_node: usize,
    /// which impurity method should be used
    pub set_impurity: F,
    /// the minimum size of data a interior node should be trained with (otherwise a leaf will be produce)
    pub min_subset_size: usize,
    /// maximal depth of this tree
    pub max_depth: usize,
}

impl<F> NumericTreeFunc<F>
where
    F: ImpurityFunc,
{
    /// Generating a new NumericTreeFunc
    /// `low_limit` is the lowest value expected for training,
    /// `high_limit` is the greatest one.
    /// `min_subset_size` the minimum size a interior node be trained with.
    /// `max_depth` the maximal depth of the tree.
    fn new(
        f: F,
        low_limit: f64,
        high_limit: f64,
        min_subset_size: usize,
        max_depth: usize,
    ) -> NumericTreeFunc<F> {
        NumericTreeFunc {
            min_nr: low_limit,
            max_nr: high_limit,
            nr_feat_per_node: 20,
            set_impurity: f,
            min_subset_size: min_subset_size,
            max_depth: max_depth,
        }
    }
}


impl<F> TreeFunction for NumericTreeFunc<F>
where
    F: ImpurityFunc,
{
    type Data = f64;
    type Param = NumericNodeParam;

    fn binarize(&self, param: &NumericNodeParam, element: &f64) -> Binar {
        if *element < param.threshold {
            Binar::Zero
        } else {
            Binar::One
        }
    }
}

/// Describes the training of a numeric tree
impl<F> TreeLearnFunctions for NumericTreeFunc<F>
where
    F: ImpurityFunc,
{
    type LeafParam = NumericLeafParam;
    type Truth = NumericTruth;
    type ParamIter = IntoIter<Self::Param>;

    /// Impurity of two sets use [ImpurityFunc](trait.ImpurityFunc.html).
    fn impurity(
        &self,
        _: &NumericNodeParam,
        set_l: &[(&f64, &NumericTruth)],
        set_r: &[(&f64, &NumericTruth)],
        _: usize,
    ) -> f64 {
        let size = (set_l.len() + set_r.len()) as f64;
        let res = ((set_l.len() as f64) * self.set_impurity.impurity(set_l) +
                       (set_r.len() as f64) * self.set_impurity.impurity(set_r)) /
            size;
        res
    }

    /// Generates some random thresholds
    fn param_set(&self) -> Self::ParamIter {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let vec: Vec<NumericNodeParam> = (0..self.nr_feat_per_node)
            .map(|_| {
                NumericNodeParam {
                    threshold: {
                        let res = rng.gen_range(self.min_nr, self.max_nr);
                        res
                    },
                }
            })
            .collect();
        vec.into_iter()
    }

    /// Computes the leaf data (probability) using the rate of InClass Truth.
    fn comp_leaf_data(&self, set: &[(&f64, &NumericTruth)]) -> NumericLeafParam {
        let size = set.len();
        let truth = set.iter().fold(0, |sum, &(_, t)| match *t {
            NumericTruth::InClass => sum + 1,
            _ => sum,
        });

        let prob = (truth as f64) / (size as f64);
        return NumericLeafParam { probability: prob };
    }

    /// Stops building an interior  leaf if `depth` is greater than wanted,
    /// `elements` only contains InClass or NotInClass members or
    /// if `elements` is not great enough.
    fn early_stop(&self, depth: usize, elements: &[(&Self::Data, &Self::Truth)]) -> bool {
        // if depth >= maxdepth || elements.len() >= max_elements
        if elements.is_empty() {
            unimplemented!()
        };
        if depth >= self.max_depth || elements.len() < self.min_subset_size {
            return true;
        }
        let inclass = elements[0].1.in_class();
        elements.iter().all(|&(_, t)| t.in_class() == inclass)
    }

    /// Use Self as TreeFunction
    fn as_predict_learn_func(self) -> Self {
        self
    }
}


pub type NumericTree = DecisionTree<NumericLeafParam, NumericTreeFunc<EntropyImpurity>>;

/// Struct for training a numeric tree
#[derive(Serialize, Deserialize)]
pub struct NumericTreeLearnParams {
    /// NumericTree specific TreeFunction
    pub func: NumericTreeFunc<EntropyImpurity>,
    /// Parameter for learning a decision tree
    pub tree_param: TreeParameters,
}

impl NumericTreeLearnParams {
    /// Creates a new NumericTreeLearnParams.
    /// `low_limit` is the lowest value expected for training,
    /// `high_limit` is the greatest one.
    /// `min_subset_size` the minimum size a interior node be trained with.
    /// `max_depth` the maximal depth of the tree.
    pub fn new(
        low_limit: f64,
        high_limit: f64,
        min_subset_size: usize,
        max_depth: usize,
    ) -> NumericTreeLearnParams {
        NumericTreeLearnParams {
            func: NumericTreeFunc::new(
                EntropyImpurity {},
                low_limit,
                high_limit,
                min_subset_size,
                max_depth,
            ),
            tree_param: TreeParameters::new(),
        }
    }

    /// Train a numeric tree using the ground truth `train_set`
    pub fn learn_tree(self, train_set: &[(f64, NumericTruth)]) -> NumericTree {
        let dataref: Vec<(&f64, &NumericTruth)> =
            train_set.iter().map(|&(ref a, ref b)| (a, b)).collect();
        let tree = self.tree_param.learn_tree(self.func, &dataref[..]);
        tree
    }
}
