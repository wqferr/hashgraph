use std::hash::Hash;
use std::collections::HashMap;

// Joins two vertices of type V, with associated edge value E
pub struct Edge<V: Hash, E> {
    source: V,
    target: V,
    value: E
}

pub struct HashGraph<V: Hash, E> {
    nodes: HashMap<V, Vec<Edge<V, E>>>
}