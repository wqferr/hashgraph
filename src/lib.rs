use std::hash::Hash;
use std::collections::HashMap;

pub struct Edge<V: Hash, E> {
    source: V,
    target: V,
    value: E
}

pub struct HashGraph<V: Hash, E> {
    nodes: HashMap<V, Edge<V, E>>
}