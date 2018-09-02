# Stamm

Stamm is a rust library for creating [decision trees](https://en.wikipedia.org/wiki/Decision_tree) and [random forests](https://en.wikipedia.org/wiki/Random_forest) in a very general way. 
Decision trees are used in machine learning for classification and regression. A random forest bundles some decision trees for making a more precise classification or regression.

This library allows to specify the data and features of the nodes within a tree and the criteria for training. Any data can be used for classification and any data can be the output of a tree. Therefore, a probability output can be used for classification and a numeric output for regression.
Furthermore, this is also true for a random forest.

To give an example, I've written a [hough forest](https://www.robots.ox.ac.uk/~vilem/cvpr2009.pdf) using this library a while ago.
I published the code [here](https://github.com/Entscheider/depthhead).

This library can use [rayon](https://github.com/rayon-rs/rayon) to parallelize the training and prediction of a random forest.
For serialization and deserialization [serde](https://github.com/serde-rs/serde) is used. So you can save the trees and random forests as json, yaml, MessagePack and in many more formats (see [here](https://serde.rs/#data-formats)).

A numeric tree implementation using Stamm can be found in the library. An example using this numeric tree is discoverable in the `examples` directory.

## Documentation

See https://docs.rs/stamm

## Usage
As always in Rust add Stamm as a dependency in Cargo.toml

```rust
[dependencies]
stamm = "*"
```

and then add to your main.rs or lib.rs


```rust
extern crate stamm;
```

If you want to create your own tree, you've to implement the `TreeLearnFunctions` trait. 
You can train your tree with something like this

```rust
let trainings_set = vec![some awesome training data];
let learn_function = MySpecialTreeLearnFunction::new();
let learner = TreeParameters::new();
let learned_tree = learner.learn_tree(learn_function, &trainings_set);
```

and use your learned tree like this 

```rust
let to_predict = some_awesome_data_to_predict;
let result = learned_tree.predict(&to_predict);
```

Training a random forest is straight forward:

```rust
let forest_learner = RandomForestLearnParam::new(
    10 /* number of trees */,
    50 /* size of the trainings subset used for a tree */,
    learn_function /* see above */);
let trained_forest = forest_learner.train_forest(&trainings_set).unwrap();
// Or if the types you are using support it - train the forest parallel
let trained_forest = forest_learner.train_forest_parallel(&trainings_set).unwrap();
```

Using it:

```rust
let result_list = 
trained_forest.forest_predictions(&to_predict);
// Or to parallelize it
let result_list = 
trained_forest.forest_predictions_parallel(&to_predict);
```

You get a vector which contains the result of every tree the forest has. You can combine them as you wish. E.g. if you want to predict, you can compute the average over all predictions to obtain a single result.

## License
Stamm is distributed under the terms of the Apache License, Version 2.0.
