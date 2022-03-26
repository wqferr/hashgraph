use std::hash::Hash;
use std::collections::{HashMap, HashSet, VecDeque, BinaryHeap, LinkedList};
use std::ops::Add;

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
#[derive(Debug)]
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
    origin: HashMap<V, &'a Edge<V, E>>
}

pub struct DepthFirstIter<'a, V: Vertex, E> {
    graph: &'a HashGraph<V, E>,
    open: VecDeque<V>,
    closed: HashSet<V>,
    origin: HashMap<V, &'a Edge<V, E>>
}

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

    pub fn breadth_first_iter(&self, start: V) -> impl Iterator<Item=(V, Option<&Edge<V, E>>)> {
        BreadthFirstIter {
            graph: self,
            closed: HashSet::new(),
            open: VecDeque::from([start]),
            origin: HashMap::new()
        }
    }

    pub fn depth_first_iter(&self, start: V) -> impl Iterator<Item=(V, Option<&Edge<V, E>>)> {
        DepthFirstIter {
            graph: self,
            closed: HashSet::new(),
            open: VecDeque::from([start]),
            origin: HashMap::new()
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

    fn size_hint(&self) -> (usize, Option<usize>) {
        (1, Some(self.graph.vertices().len()))
    }

    fn next(&mut self) -> Option<Self::Item> {
        if self.open.is_empty() {
            return None;
        }
        let current = self.open.pop_front().unwrap();
        self.closed.insert(current);
        let edge_to_current = self.origin.remove(&current);
        for edge in self.graph.edges_from(current) {
            let next = edge.target;
            if !self.closed.contains(&next) {
                self.open.push_back(next);
                self.origin.insert(next, edge);
            }
        }
        Some((current, edge_to_current))
    }
}

impl<'a, V: Vertex, E> Iterator for DepthFirstIter<'a, V, E> {
    type Item = (V, Option<&'a Edge<V, E>>);

    fn size_hint(&self) -> (usize, Option<usize>) {
        (1, Some(self.graph.vertices().len()))
    }
    
    fn next(&mut self) -> Option<Self::Item> {
        let mut current;

        loop {
            if self.open.is_empty() {
                return None;
            }
            current = self.open.pop_back().unwrap();
            if !self.closed.contains(&current) {
                break;
            }
        }

        self.closed.insert(current);
        let edge_to_current = self.origin.remove(&current);
        for edge in self.graph.edges_from(current) {
            let next = edge.target;
            if !self.closed.contains(&next) {
                self.open.push_back(next);
                self.origin.insert(next, edge);
            }
        }
        Some((current, edge_to_current))
    }
}

pub trait Cost: Sized + Add<Output=Self> + Ord + Clone + Default {}
impl<T: Sized + Add<Output=Self> + Ord + Clone + Default> Cost for T {}

#[derive(Eq, Debug)]
struct PrioritizedVertex<V: Vertex, C: Cost>(V, C);

impl<V: Vertex, C: Cost> PartialEq for PrioritizedVertex<V, C> {
    fn eq(&self, other: &Self) -> bool {
        self.1 == other.1
    }
}

impl<V: Vertex, C: Cost> PartialOrd for PrioritizedVertex<V, C> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<V: Vertex, C: Cost> Ord for PrioritizedVertex<V, C> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match self.1.cmp(&other.1) {
            // Invert order for max-heap usage
            std::cmp::Ordering::Less => std::cmp::Ordering::Greater,
            std::cmp::Ordering::Equal => std::cmp::Ordering::Equal,
            std::cmp::Ordering::Greater => std::cmp::Ordering::Less,
        }
    }
}

impl<V: Vertex, E: Cost> HashGraph<V, E> {
    // This *will* repeatedly clone edge costs
    pub fn min_cost_path(&self, start: V, end: V) -> Option<(E, Vec<&Edge<V, E>>)> {
        let mut origin = HashMap::new();
        let mut discovered_cost = HashMap::new();
        let mut open = BinaryHeap::<PrioritizedVertex<V, E>>::new();
        let mut closed = HashSet::new();

        open.push(PrioritizedVertex(start, E::default()));

        while let Some(PrioritizedVertex(current, current_cost)) = open.pop() {
            closed.insert(current);
            if current == end {
                break;
            }

            for edge in self.edges_from(current) {
                let neighbor = edge.target();
                if closed.contains(neighbor) {
                    continue;
                }

                let new_total_cost = current_cost.clone() + edge.value().to_owned();
                if let Some(prev_total_cost) = discovered_cost.get(neighbor) {
                    if &new_total_cost < prev_total_cost {
                        open.push(PrioritizedVertex(neighbor.to_owned(), new_total_cost.to_owned()));
                        discovered_cost.insert(neighbor.to_owned(), new_total_cost);
                        origin.insert(neighbor.to_owned(), edge);
                    }
                } else {
                    open.push(PrioritizedVertex(neighbor.to_owned(), new_total_cost.to_owned()));
                    discovered_cost.insert(neighbor.to_owned(), new_total_cost);
                    origin.insert(neighbor.to_owned(), edge);
                }
            }
        }

        if let Some(cost) = discovered_cost.get(&end) {
            let mut path = LinkedList::new();
            let mut current = *origin.get(&end).unwrap();
            while current.source() != &start {
                path.push_front(current);
                current = origin.get(current.source()).unwrap();
            }
            path.push_front(current);
            Some((cost.to_owned(), path.into_iter().collect()))
        } else {
            None
        }
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
    fn can_do_breadth_first_search() {
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
        for (i, (vertex, edge)) in g.breadth_first_iter('a').enumerate() {
            if let Some(edge) = edge {
                assert_eq!(&vertex, edge.target());
            }
            visit_order.insert(vertex, i);
        }

        assert!(visit_order[&'a'] == 0);
        assert!(visit_order[&'b'] < visit_order[&'d']);
        assert!(visit_order[&'c'] < visit_order[&'d']);
        assert!(visit_order[&'d'] < visit_order[&'e']);
        assert!(!visit_order.contains_key(&'f'));
    }

    #[test]
    fn can_do_depth_first_search() {
        let mut g = HashGraph::with_vertices(['a', 'b', 'c', 'd', 'e', 'f']).unwrap();
        g.create_edge('a', 'b', 2.0).unwrap();
        g.create_edge('a', 'c', 3.0).unwrap();
        g.create_edge('b', 'd', 5.0).unwrap();
        g.create_edge('c', 'd', 1.0).unwrap();
        g.create_edge('d', 'e', 2.0).unwrap();

        g.create_edge('f', 'c', 1.0).unwrap();
        g.create_edge('c', 'a', 5.0).unwrap();
        // 'f' can't be reached

        let mut visited = HashSet::new();
        for (vertex, edge) in g.depth_first_iter('a') {
            if let Some(edge) = edge {
                assert_eq!(&vertex, edge.target());
            }
            visited.insert(vertex);
            assert!(vertex != 'f', "reached isolated vertex");
        }

        for v in g.vertices() {
            if v != &'f' {
                // Reached everything else
                assert!(visited.contains(v), "didn't reach node {}", v);
            }
        }
    }
}

#[cfg(test)]
mod pathfinding_tests {
    use super::*;

    #[test]
    fn test_dijkstra() {
        // Image from https://www.chegg.com/homework-help/questions-and-answers/8-4-14-10-2-figure-2-directed-graph-computing-shortest-path-3-dijkstra-s-algorithm-computi-q25960616
        let mut g = HashGraph::with_vertices(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']).unwrap();
        g.create_edge('a', 'b', 4).unwrap();

        g.create_edge('b', 'c', 11).unwrap();
        g.create_edge('b', 'd', 9).unwrap();

        g.create_edge('c', 'a', 8).unwrap();

        g.create_edge('d', 'c', 7).unwrap();
        g.create_edge('d', 'e', 2).unwrap();
        g.create_edge('d', 'f', 6).unwrap();

        g.create_edge('e', 'b', 8).unwrap();
        g.create_edge('e', 'g', 7).unwrap();
        g.create_edge('e', 'h', 4).unwrap();

        g.create_edge('f', 'c', 1).unwrap();
        g.create_edge('f', 'e', 5).unwrap();

        g.create_edge('g', 'h', 14).unwrap();
        g.create_edge('g', 'i', 9).unwrap();

        g.create_edge('h', 'f', 2).unwrap();
        g.create_edge('h', 'i', 10).unwrap();

        let result = g.min_cost_path('b', 'i');
        assert!(result.is_some());
        let (cost, path) = result.unwrap();
        assert_eq!(25, cost);
        assert_eq!(&'b', path[0].source());
        assert_eq!(
            vec!['d', 'e', 'h', 'i'],
            path.iter()
                .map(|e| e.target().to_owned())
                .collect::<Vec<_>>()
        );
    }
}