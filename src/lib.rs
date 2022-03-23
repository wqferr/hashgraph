use std::hash::Hash;
use std::collections::{HashMap, HashSet, VecDeque};

#[derive(Debug, PartialEq, Eq)]
pub enum Error {
    NonexistentVertex,
    DuplicateVertex,
    NonexistentEdge,
    DuplicateEdge
}

pub type Result<T> = std::result::Result<T, Error>;

pub trait Vertex: Hash + Eq + Copy {}
impl<T: Hash + Eq + Copy> Vertex for T {}

// Joins two vertices of type V, with associated edge value E
pub struct Edge<V: Vertex, E> {
    source: V,
    target: V,
    value: E
}

// Does not allow for more than 1 edge per pair of vertices
pub struct HashGraph<V: Vertex, E> {
    // Set of existing nodes
    nodes: HashSet<V>,

    // Maps each node to its outgoing edges
    node_edges: HashMap<V, HashSet<V>>,

    // Maps pairs of nodes to their connecting edge, if it exists
    edges: HashMap<(V, V), Edge<V, E>>
}

pub struct BreadthFirstIter<'a, V: Vertex, E> {
    graph: &'a HashGraph<V, E>,
    open: VecDeque<V>,
    closed: HashSet<V>,
    parent: HashMap<V, &'a Edge<V, E>>
}

// TODO
// pub struct DepthFirstIter<'a, V: Vertex, E> {
//     graph: &'a HashGraph<V, E>,
//     current: V,
//     visited: HashSet<V>
// }

impl<V: Vertex, E> Edge<V, E> {
    pub fn source(&self) -> &V {
        &self.source
    }

    pub fn target(&self) -> &V {
        &self.target
    }

    pub fn value(&self) -> &E {
        &self.value
    }
}

impl<V: Vertex, E> HashGraph<V, E> {
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
        if self.has_vertex(vertex) {
            Err(Error::DuplicateVertex)
        } else {
            self.create_vertex_unsafe(vertex);
            Ok(())
        }
    }

    pub fn has_vertex(&self, vertex: V) -> bool {
        self.nodes.contains(&vertex)
    }

    // Errors if any of the given vertices would've been duplicates.
    // Will not add any vertices if there is an error.
    pub fn create_vertices(&mut self, vertices: Vec<V>) -> Result<()> {
        for &v in vertices.iter() {
            if self.has_vertex(v) {
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

    pub fn create_edge(&mut self, source: V, target: V, value: E) -> Result<()> {
        if !self.has_vertex(source) || !self.has_vertex(target) {
            return Err(Error::NonexistentVertex);
        } else if self.has_edge(source, target) {
            return Err(Error::DuplicateEdge);
        }

        self.create_edge_unsafe(source, target, value);
        Ok(())
    }

    pub fn drop_edge(&mut self, source: V, target: V) -> Result<()> {
        if !self.has_edge(source, target) {
            return Err(Error::NonexistentEdge);
        }

        // This will panic in case both fields get desynced somehow
        self.node_edges.get_mut(&source).unwrap().remove(&target);
        self.edges.remove(&(source, target));
        Ok(())
    }

    pub fn has_edge(&self, source: V, target: V) -> bool {
        self.edges.contains_key(&(source, target))
    }

    pub fn edges_from(&self, source: V) -> Vec<&Edge<V, E>> {
        let targets = &self.node_edges[&source];
        let mut edges = vec![];
        for &t in targets {
            let e = &self.edges[&(source, t)];
            edges.push(e);
        }
        edges
    }

    pub fn edges(&self) -> Vec<&Edge<V, E>> {
        self.edges.values().collect()
    }

    pub fn broad_iter(&self, start: V) -> impl Iterator<Item=(V, Option<&Edge<V, E>>)> {
        BreadthFirstIter {
            closed: HashSet::new(),
            open: VecDeque::from([start]),
            graph: self,
            parent: HashMap::new()
        }
    }

    fn create_vertex_unsafe(&mut self, vertex: V) {
        self.nodes.insert(vertex.clone());
        self.node_edges.insert(vertex, HashSet::new());
    }

    fn create_edge_unsafe(&mut self, source: V, target: V, value: E) {
        let new_edge = Edge { source: source, target: target, value: value };
        self.edges.insert((source, target), new_edge);
        self.node_edges.get_mut(&source).unwrap().insert(target);
    }
}

impl<'a, V: Vertex, E> Iterator for BreadthFirstIter<'a, V, E> {
    type Item = (V, Option<&'a Edge<V, E>>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.open.is_empty() {
            return None;
        }
        let current = self.open.pop_front().unwrap();
        self.closed.insert(current);
        let edge_to_current = self.parent.remove(&current);
        for edge in self.graph.edges_from(current) {
            let next = edge.target;
            if !self.closed.contains(&next) {
                self.open.push_back(next);
                self.parent.insert(next, edge);
            }
        }
        Some((current, edge_to_current))
    }
}

#[cfg(test)]
mod vertex_tests {
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

        assert!(g.create_vertex('a').is_ok());
        expected_vertices.insert('a');
        assert!(g.has_vertex('a'));
        assert_eq!(&expected_vertices, g.vertices());

        assert!(g.create_vertex('b').is_ok());
        expected_vertices.insert('b');
        assert!(g.has_vertex('b'));
        assert_eq!(&expected_vertices, g.vertices());
    }

    #[test]
    fn can_detect_duplicate_vertices() {
        let mut g = HashGraph::<char, f32>::new();
        assert!(g.create_vertex('a').is_ok());
        assert_eq!(Err(Error::DuplicateVertex), g.create_vertex('a'));
    }

    #[test]
    fn can_create_populated() {
        let g = HashGraph::<char, f32>::with_vertices(['a', 'b', 'c']).unwrap();
        let expected_vertices = HashSet::from(['a', 'b', 'c']);
        assert_eq!(&expected_vertices, g.vertices());
    }
}


#[cfg(test)]
mod edge_tests {
    use super::*;

    #[test]
    fn can_create_edges() {
        let mut g = HashGraph::with_vertices(['a', 'b', 'c']).unwrap();
        assert!(g.create_edge('a', 'b', 5.0).is_ok());
        assert!(g.create_edge('a', 'c', 3.0).is_ok());
        assert!(g.create_edge('b', 'c', 2.0).is_ok());

        let mut edges_from_a = g.edges_from('a');
        let edges_from_b = g.edges_from('b');
        let edges_from_c = g.edges_from('c');
        assert_eq!(2, edges_from_a.len());
        assert_eq!(1, edges_from_b.len());
        assert_eq!(0, edges_from_c.len());

        // Make sure 'a'-'b' is edge #0
        edges_from_a.sort_by(|&x, &y| x.target().cmp(y.target()));

        assert_eq!(&'a', edges_from_a[0].source());
        assert_eq!(&'b', edges_from_a[0].target());
        assert_eq!(&5.0, edges_from_a[0].value());

        assert_eq!(&'a', edges_from_a[1].source());
        assert_eq!(&'c', edges_from_a[1].target());
        assert_eq!(&3.0, edges_from_a[1].value());
        
        assert_eq!(&'b', edges_from_b[0].source());
        assert_eq!(&'c', edges_from_b[0].target());
        assert_eq!(&2.0, edges_from_b[0].value());
    }


    #[test]
    fn can_detect_duplicate_edges() {
        let mut g = HashGraph::with_vertices(['a', 'b', 'c']).unwrap();
        assert!(g.create_edge('a', 'b', 5.0).is_ok());
        assert_eq!(Err(Error::DuplicateEdge), g.create_edge('a', 'b', 3.0));
        assert!(g.create_edge('b', 'a', 5.0).is_ok());
    }

    #[test]
    fn can_detect_edge_between_nonexistent_vertices() {
        let mut g = HashGraph::with_vertices(['a', 'b', 'c']).unwrap();
        assert_eq!(Err(Error::NonexistentVertex), g.create_edge('a', 'd', 2.0));
        assert_eq!(Err(Error::NonexistentVertex), g.create_edge('d', 'a', 2.0));
        assert_eq!(Err(Error::NonexistentVertex), g.create_edge('d', 'e', 2.0));
    }

    #[test]
    fn can_drop_edges() {
        let mut g = HashGraph::with_vertices(['a', 'b', 'c']).unwrap();
        g.create_edge('a', 'b', 3.0).unwrap();
        g.create_edge('a', 'c', 1.0).unwrap();
        
        assert!(g.drop_edge('a', 'b').is_ok());
        let edges = g.edges_from('a');
        assert_eq!(1, edges.len());
        assert_eq!(&'c', edges[0].target());
    }

    #[test]
    fn can_detect_unknown_edges() {
        let mut g = HashGraph::with_vertices(['a', 'b', 'c']).unwrap();
        g.create_edge('a', 'b', 3.0).unwrap();
        g.create_edge('a', 'c', 1.0).unwrap();

        assert_eq!(Err(Error::NonexistentEdge), g.drop_edge('b', 'a'));
        assert_eq!(Err(Error::NonexistentEdge), g.drop_edge('c', 'b'));
    }
}

#[cfg(test)]
mod iter_tests {
    use super::*;

    #[test]
    fn can_do_breadth_first_traversal() {
        let mut g = HashGraph::with_vertices(['a', 'b', 'c', 'd', 'e', 'f']).unwrap();
        g.create_edge('a', 'b', 2.0).unwrap();
        g.create_edge('a', 'c', 3.0).unwrap();
        g.create_edge('b', 'd', 5.0).unwrap();
        g.create_edge('c', 'd', 1.0).unwrap();
        g.create_edge('d', 'e', 2.0).unwrap();

        g.create_edge('f', 'c', 1.0).unwrap();
        g.create_edge('c', 'a', 5.0).unwrap();
        // 'f' can't be reached

        let mut visit_order = HashMap::<char, usize>::new();
        for (i, (vertex, _edge)) in g.broad_iter('a').enumerate() {
            visit_order.insert(vertex, i);
        }

        assert!(visit_order[&'a'] == 0);
        assert!(visit_order[&'b'] < visit_order[&'d']);
        assert!(visit_order[&'c'] < visit_order[&'d']);
        assert!(visit_order[&'d'] < visit_order[&'e']);
        assert!(!visit_order.contains_key(&'f'));
    }
}