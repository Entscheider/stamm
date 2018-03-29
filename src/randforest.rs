//! Implements a random forest using `stamm::tree`.
use super::tree::*;
use rand;
use rand::Rng;
use serde::Serialize;
use serde::de::DeserializeOwned;
use rayon::prelude::*;

/// Combine the results of all trees in a random forest  in a numeric way
pub trait VotingMethod<L> {
    fn voting(&self, tree_results: &[&L]) -> f64;
}

/// For voting some probability this trait should be implemented by the leaf data for extracting this probability
pub trait FromGetProbability {
    fn probability(&self) -> f64;
}

impl<T> FromGetProbability for T
where
    f64: From<T>,
    T: Copy,
{
    fn probability(&self) -> f64 {
        return f64::from(*self);
    }
}

/// Implementing average voting
pub struct AverageVoting;
impl<L> VotingMethod<L> for AverageVoting
where
    L: FromGetProbability,
{
    fn voting(&self, tree_results: &[&L]) -> f64 {
        let sum = tree_results.iter().fold(
            0f64,
            |sum, l| sum + l.probability(),
        );
        return (sum as f64) / (tree_results.len() as f64);
    }
}

/// A random forest to combine some decision trees.
/// The decision trees have leafs with data from type `L`
/// and use a TreeFunction from type `F`.
#[derive(Serialize, Deserialize)]
#[serde(bound(serialize = "DecisionTree<L,F>: Serialize"))]
#[serde(bound(deserialize = "DecisionTree<L,F>: DeserializeOwned"))]
pub struct RandomForest<L, F>
where
    F: TreeFunction,
{
    subtrees: Vec<DecisionTree<L, F>>,
}

impl<L, F> RandomForest<L, F>
where
    F: TreeFunction,
{
    /// Let every tree predict a result of the `input`.
    /// Returns the results in form a Vec of the leaf data.
    pub fn forest_predictions(&self, input: &F::Data) -> Vec<&L> {
        self.subtrees
            .iter()
            .filter_map(|tree| tree.predict(input))
            .collect()
    }

    /// Let every tree predict a result and combine them using a vote method.
    pub fn predict<V>(&self, input: &F::Data, voting_method: V) -> Option<f64>
    where
        V: VotingMethod<L>,
    {
        let predictions: Vec<_> = self.forest_predictions(input);
        Some(voting_method.voting(&predictions[..]))
    }
}

impl<L, F> RandomForest<L, F>
where
    F: TreeFunction + Send + Sync,
    <F as TreeFunction>::Param: Send + Sync,
    <F as TreeFunction>::Data: Send + Sync,
    L: Send + Sync,
{
    /// Like [`forest_predictions`](#method.forest_predictions)
    /// but use rayon to parallelize the computation.
    pub fn forest_predictions_parallel(&self, input: &F::Data) -> Vec<&L> {
        self.subtrees
            .par_iter()
            .filter_map(|tree| tree.predict(input))
            .fold(|| Vec::with_capacity(self.subtrees.len()), |mut v, x| {
                v.push(x);
                v
            })
            .reduce(|| Vec::with_capacity(self.subtrees.len()), |mut v, mut x| {
                v.append(&mut x);
                v
            })
    }
}


/// Parameter describes the way to train a random forest
pub struct RandomForestLearnParam<LearnF>
where
    LearnF: TreeLearnFunctions,
{
    /// parameter used for every tree
    pub tree_param: TreeParameters,
    /// number of trees
    pub number_of_trees: usize,
    /// size of a random training subset used for train one tree
    pub size_of_subset_per_training: usize,
    /// TreeLearnFunction
    pub learn_function: LearnF,
}

impl<LearnF> RandomForestLearnParam<LearnF>
where
    LearnF: TreeLearnFunctions + Copy,
{
    /// Creates a new RandomForestLearnParam.
    /// `number_of_trees` is the number of trees used in this random forest.
    /// Every tree will be trained using a random subset of the training data. `size_of_subset_per_training` is the size of this subset.
    /// `learnf` is the TreeLearnFunction for every tree
    pub fn new(
        number_of_trees: usize,
        size_of_subset_per_training: usize,
        learnf: LearnF,
    ) -> RandomForestLearnParam<LearnF> {
        RandomForestLearnParam {
            tree_param: TreeParameters::new(),
            number_of_trees: number_of_trees,
            size_of_subset_per_training: size_of_subset_per_training,
            learn_function: learnf,
        }
    }

    /// Trains a random forest using the ground truth data `train_set`.
    pub fn train_forest(
        self,
        train_set: &[(&LearnF::Data, &LearnF::Truth)],
    ) -> Option<RandomForest<LearnF::LeafParam, LearnF::PredictFunction>> {
        let mut res = vec![];
        let mut rng = rand::thread_rng();
        let mut subset = Vec::with_capacity(self.size_of_subset_per_training);
        for _ in 0..self.number_of_trees {
            subset.clear();
            for _ in 0..self.size_of_subset_per_training {
                subset.push(train_set[rng.gen_range(0, train_set.len())]);
            }
            let tree = self.tree_param.learn_tree(self.learn_function, &subset[..]);
            res.push(tree);
        }
        Some(RandomForest { subtrees: res })

    }
}
impl<LearnF> RandomForestLearnParam<LearnF>
where
    LearnF: TreeLearnFunctions + Copy + Send + Sync,
    LearnF::PredictFunction: Send + Sync,
    LearnF::Truth: Send + Sync,
    LearnF::LeafParam: Send + Sync,
    LearnF::Data: Send + Sync,
    LearnF::Param: Send + Sync,
{
    /// Like [`train_forest`](#method.train_forest)
    /// but use rayon to parallelize the training.
    pub fn train_forest_parallel(
        self,
        train_set: &[(&LearnF::Data, &LearnF::Truth)],
    ) -> Option<RandomForest<LearnF::LeafParam, LearnF::PredictFunction>> {

        let subset_size = self.size_of_subset_per_training;
        let trees = (0..self.number_of_trees)
            .into_par_iter()
            .map(|_| {
                let mut rng = rand::thread_rng();
                let mut subset = Vec::with_capacity(self.size_of_subset_per_training);
                for _ in 0..subset_size {
                    subset.push(train_set[rng.gen_range(0, train_set.len())]);
                }
                let tree = self.tree_param.learn_tree(self.learn_function, &subset[..]);
                tree
            })
            .fold(|| Vec::with_capacity(subset_size), |mut v, x| {
                v.push(x);
                v
            })
            .reduce(|| Vec::with_capacity(subset_size), |mut v, mut x| {
                v.append(&mut x);
                v
            });
        Some(RandomForest { subtrees: trees })
    }
}
