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

// Does not allow for more than 1 edge per pair of vertices
pub struct HashGraph<V: Hash + Eq + Copy, E> {
    nodes: HashSet<V>,
    node_edges: HashMap<V, Vec<Edge<V, E>>>,
    edges: HashMap<(V, V), Edge<V, E>>
}

impl<V: Hash + Eq + Copy, E> HashGraph<V, E> {
    pub fn new() -> Self {
        Self {
            nodes: HashSet::new(),
            node_edges: HashMap::new(),
            edges: HashMap::new()
        }
    }

    pub fn with_vertices<const N: usize>(vertices: [V; N]) -> Result<Self> {
        let mut graph = HashGraph::new();
        for v in vertices {
            graph.create_vertex(v)?;
        }
        Ok(graph)
    }

    pub fn create_vertex(&mut self, vertex: V) -> Result<()> {
        if self.has_vertex(&vertex) {
            Err(Error::DuplicateVertex)
        } else {
            self.create_vertex_unsafe(vertex);
            Ok(())
        }
    }

    pub fn has_vertex(&self, vertex: &V) -> bool {
        self.nodes.contains(vertex)
    }

    // Errors if any of the given vertices would've been duplicates.
    // Will not add any vertices if there is an error.
    pub fn create_vertices(&mut self, vertices: Vec<V>) -> Result<()> {
        for v in vertices.iter() {
            if self.has_vertex(&v) {
                return Err(Error::DuplicateVertex)
            }
        }

        // Separate loop so it only changes self if there were no errors
        for v in vertices {
            self.create_vertex_unsafe(v);
        }
        Ok(())
    }

    pub fn vertices(&self) -> &HashSet<V> {
        &self.nodes
    }

    fn create_vertex_unsafe(&mut self, vertex: V) {
        self.nodes.insert(vertex.clone());
        self.node_edges.insert(vertex, vec![]);
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn can_create_empty_graph() {
        let g = HashGraph::<char, f32>::new();
        assert_eq!(&HashSet::<char>::new(), g.vertices());
    }

    #[test]
    fn can_add_vertices() {
        let mut g = HashGraph::<char, f32>::new();
        let mut expected_vertices = HashSet::<char>::new();

        assert_eq!(Ok(()), g.create_vertex('a'));
        expected_vertices.insert('a');
        assert!(g.has_vertex(&'a'));
        assert_eq!(&expected_vertices, g.vertices());

        assert_eq!(Ok(()), g.create_vertex('b'));
        expected_vertices.insert('b');
        assert!(g.has_vertex(&'b'));
        assert_eq!(&expected_vertices, g.vertices());
    }

    #[test]
    fn can_detect_duplicate_vertices() {
        let mut g = HashGraph::<char, f32>::new();
        assert_eq!(Ok(()), g.create_vertex('a'));
        assert_eq!(Err(Error::DuplicateVertex), g.create_vertex('a'));
    }

    #[test]
    fn can_create_populated() {
        let g = HashGraph::<char, f32>::with_vertices(['a', 'b', 'c']).unwrap();
        let expected_vertices = HashSet::from([&'a', &'b', &'c']);
        // assert_eq!(expected_vertices, g.vertices());
    }
}