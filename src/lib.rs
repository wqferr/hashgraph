use std::hash::Hash;
use std::collections::{HashMap, HashSet};

#[derive(Debug, PartialEq, Eq)]
pub enum Error {
    DuplicateVertex
}

pub type Result<T> = std::result::Result<T, Error>;

// Joins two vertices of type V, with associated edge value E
pub struct Edge<V: Hash + Eq, E> {
    source: V,
    target: V,
    value: E
}

pub struct HashGraph<'a, V: Hash + Eq, E> {
    nodes: HashMap<V, Vec<&'a Edge<V, E>>>,
    edges: HashMap<(V, V), E>
}

impl<'a, V: Hash + Eq, E> HashGraph<'a, V, E> {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            edges: HashMap::new()
        }
    }

    pub fn with_vertices(vertices: Vec<V>) -> Result<Self> {
        let mut graph = HashGraph::new();
        for v in vertices {
            graph.create_vertex(v)?;
        }
        Ok(graph)
    }

    pub fn create_vertex(&mut self, vertex: V) -> Result<()> {
        if self.nodes.contains_key(&vertex) {
            Err(Error::DuplicateVertex)
        } else {
            self.nodes.insert(vertex, vec![]);
            Ok(())
        }
    }

    pub fn vertices(&self) -> HashSet<&V> {
        self.nodes.keys().collect()
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn can_create_empty_graph() {
        let g = HashGraph::<char, f32>::new();
        assert_eq!(HashSet::<&char>::new(), g.vertices());
    }

    #[test]
    fn can_add_vertices() {
        let mut g = HashGraph::<char, f32>::new();
        let mut expected_vertices = HashSet::<&char>::new();

        assert_eq!(Ok(()), g.create_vertex('a'));
        expected_vertices.insert(&'a');
        assert_eq!(expected_vertices, g.vertices());

        assert_eq!(Ok(()), g.create_vertex('b'));
        expected_vertices.insert(&'b');
        assert_eq!(expected_vertices, g.vertices());
    }

    #[test]
    fn can_detect_duplicate_vertices() {
        let mut g = HashGraph::<char, f32>::new();
        assert_eq!(Ok(()), g.create_vertex('a'));
        assert_eq!(Err(Error::DuplicateVertex), g.create_vertex('a'));
    }
}