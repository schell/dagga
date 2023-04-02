//! Support for serializing [`Dag`](super::Dag) as a dot file, for use with
//! graphiz.
use std::any::Any;

use super::*;

#[derive(Debug, Snafu)]
pub enum DotError {
    #[snafu(display("Could not create file: {}", source))]
    CreateFile { source: std::io::Error },

    #[snafu(display("{}", source))]
    Dot { source: dot2::Error },
}

const GHOST_ROOT_NAME: &str = r#"required_resources"#;

/// A `Dag` and some labels for providing user-facing names for resources.
pub struct DagLegend<E> {
    pub resource_labels: FxHashMap<E, String>,
    pub node_ids: FxHashMap<String, usize>,
    pub name: String,
    pub dag: Dag<(), E>,
    pub schedule: Schedule<Node<(), E>>,
    pub root: Option<Node<(), E>>,
}

impl<E: Any + Copy + PartialEq + Eq + std::hash::Hash> DagLegend<E> {
    pub fn new<'a, N: Any>(nodes: impl Iterator<Item = &'a Node<N, E>>) -> Self {
        let resource_labels = FxHashMap::default();
        let mut node_ids = FxHashMap::default();
        let mut new_dag = Dag::default();
        let mut next_id = 0;
        for node in nodes {
            if !node_ids.contains_key(node.name()) {
                node_ids.insert(node.name().to_string(), next_id);
                next_id += 1;
            }
            new_dag.add_node(node_to_dot_node(node));
        }
        let schedule = new_dag.clone().build_schedule().unwrap();
        let missing_inputs = new_dag.get_missing_inputs();
        let root = if missing_inputs.is_empty() {
            None
        } else {
            Some({
                let mut node = Node::new(())
                .with_results(missing_inputs.into_iter());
                node.name = GHOST_ROOT_NAME.to_string();
                node
            })
        };
        DagLegend {
            resource_labels,
            node_ids,
            name: String::new(),
            dag: new_dag,
            schedule,
            root,
        }
    }

    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    pub fn with_resources_named(mut self, f: impl Fn(&E) -> String) -> Self {
        for node in self.dag.nodes() {
            let resources = node.all_inputs().into_iter().chain(node.results.iter().copied());
            for resource in resources {
                if self.resource_labels.contains_key(&resource) {
                    continue;
                }
                let name = f(&resource);
                self.resource_labels.insert(resource, name);
            }
        }
        self
    }

    pub fn save_to(self, path: impl AsRef<std::path::Path>) -> Result<(), DotError> {
        save_as_dot(&self, path)
    }
}

impl<E: Any + Copy + PartialEq + Eq + std::hash::Hash + std::fmt::Debug> DagLegend<E> {
    pub fn with_resource_debug_names(self) -> Self {
        self.with_resources_named(|r| format!("{r:?}"))
    }
}

impl<E: Any + Copy + PartialEq + Eq + std::hash::Hash + std::fmt::Display> DagLegend<E> {
    pub fn with_resource_display_names(self) -> Self {
        self.with_resources_named(|r| format!("{r}"))
    }
}

pub fn save_as_dot<E: Copy + PartialEq + Eq + std::hash::Hash>(
    legend: &DagLegend<E>,
    path: impl AsRef<std::path::Path>,
) -> Result<(), DotError> {
    let mut file = std::fs::File::create(path).context(CreateFileSnafu)?;
    dot2::render(legend, &mut file).context(DotSnafu)
}

fn node_to_dot_node<T, E: Clone>(node: &Node<T, E>) -> Node<(), E> {
    let Node {
        name,
        barrier,
        moves,
        reads,
        writes,
        results,
        run_before,
        run_after,
        ..
    } = node;
    Node {
        node: (),
        name: name.clone(),
        barrier: barrier.clone(),
        moves: moves.clone(),
        reads: reads.clone(),
        writes: writes.clone(),
        results: results.clone(),
        run_before: run_before.clone(),
        run_after: run_after.clone(),
    }
}

#[derive(Clone)]
pub struct Edge<T> {
    rez: T,
    node: String,
}

impl<'a, E: Copy + PartialEq + Eq + std::hash::Hash> dot2::Labeller<'a> for DagLegend<E> {
    type Node = Node<(), E>;

    type Edge = Edge<E>;

    type Subgraph = usize;

    fn graph_id(&'a self) -> dot2::Result<dot2::Id<'a>> {
        if self.name.is_empty() {
            dot2::Id::new("my_graph")
        } else {
            dot2::Id::new(&self.name)
        }
    }

    fn node_id(&'a self, n: &Self::Node) -> dot2::Result<dot2::Id<'a>> {
        println!("name: {}", n.name);
        let id = self.node_ids.get(&n.name).map(|id| format!("node_{id}")).unwrap_or_else(|| "ghost_node".to_string());
        println!("id:{id}");
        Ok(dot2::Id::new(id).unwrap())
    }

    fn edge_label(&'a self, e: &Self::Edge) -> dot2::label::Text<'a> {
        dot2::label::Text::LabelStr(
            self.resource_labels
                .get(&e.rez)
                .map(|s| s.to_string())
                .unwrap_or_default()
                .into(),
        )
    }

    fn edge_color(&'a self, e: &Self::Edge) -> Option<dot2::label::Text<'a>> {
        let input_node = self.dag.get_node(&e.node)?;
        let color = if input_node.reads.contains(&e.rez) {
            "limegreen"
        } else if input_node.writes.contains(&e.rez) {
            "mediumblue"
        } else if input_node.moves.contains(&e.rez) {
            "tomato"
        } else {
            return None;
        };
        Some(dot2::label::Text::LabelStr(color.into()))
    }

    fn subgraph_id(&'a self, batch_index: &Self::Subgraph) -> Option<dot2::Id<'a>> {
        Some(dot2::Id::new(format!("cluster_{batch_index}")).unwrap())
    }

    fn subgraph_label(&'a self, batch_index: &Self::Subgraph) -> dot2::label::Text<'a> {
        dot2::label::Text::LabelStr(format!("batch_{batch_index}").into())
    }

    fn node_label(&'a self, n: &Self::Node) -> dot2::Result<dot2::label::Text<'a>> {
        Ok(dot2::label::Text::LabelStr(n.name.clone().into()))
    }

    fn node_shape(&'a self, _node: &Self::Node) -> Option<dot2::label::Text<'a>> {
        None
    }

    fn node_style(&'a self, n: &Self::Node) -> dot2::Style {
        println!("{}", n.name);
        if n.name.as_str() == GHOST_ROOT_NAME {
            dot2::Style::Dotted
        } else {
            dot2::Style::None
        }
    }

    fn node_color(&'a self, _node: &Self::Node) -> Option<dot2::label::Text<'a>> {
        None
    }

    fn edge_end_arrow(&'a self, _e: &Self::Edge) -> dot2::Arrow {
        dot2::Arrow::default()
    }

    fn edge_start_arrow(&'a self, _e: &Self::Edge) -> dot2::Arrow {
        dot2::Arrow::default()
    }

    fn edge_style(&'a self, _e: &Self::Edge) -> dot2::Style {
        dot2::Style::None
    }

    fn subgraph_style(&'a self, _s: &Self::Subgraph) -> dot2::Style {
        dot2::Style::None
    }

    fn subgraph_shape(&'a self, _s: &Self::Subgraph) -> Option<dot2::label::Text<'a>> {
        None
    }

    fn subgraph_color(&'a self, _s: &Self::Subgraph) -> Option<dot2::label::Text<'a>> {
        None
    }

    fn kind(&self) -> dot2::Kind {
        dot2::Kind::Digraph
    }
}

fn get_edges<T, E: Copy + PartialEq + Eq + std::hash::Hash>(dag: &Dag<T, E>, results: impl IntoIterator<Item = E>) -> Vec<Edge<E>> {
    let mut edges = vec![];
    for result in results.into_iter() {
        for downstream_node in dag.get_nodes_with_input(result) {
            edges.push(Edge {
                rez: result,
                node: downstream_node.name.clone(),
            });
        }
    }
    edges
}

impl<'a, E: Copy + PartialEq + Eq + std::hash::Hash> dot2::GraphWalk<'a> for DagLegend<E> {
    type Node = Node<(), E>;

    type Edge = Edge<E>;

    type Subgraph = usize;

    fn nodes(&'a self) -> dot2::Nodes<'a, Self::Node> {
        let mut nodes = self
            .dag
            .nodes
            .iter()
            .map(node_to_dot_node)
            .collect::<Vec<_>>();
        if let Some(root) = self.root.as_ref() {
            nodes.push(root.clone());
        }
        nodes.into()
    }

    fn edges(&'a self) -> dot2::Edges<'a, Self::Edge> {
        let mut edges: Vec<Edge<E>> = vec![];
        if let Some(root) = self.root.as_ref() {
            edges.extend(get_edges(&self.dag, root.results.clone()));
        }
        for node in self.dag.nodes.iter() {
            edges.extend(get_edges(&self.dag, node.results.clone()));
        }
        edges.into()
    }

    fn source(&'a self, edge: &Self::Edge) -> Self::Node {
        self.dag
            .get_node_that_results_in(edge.rez)
            .map(node_to_dot_node)
            .or(self.root.clone())
            .unwrap()
    }

    fn target(&'a self, edge: &Self::Edge) -> Self::Node {
        self.dag.get_node(&edge.node).map(node_to_dot_node).unwrap()
    }

    fn subgraphs(&'a self) -> dot2::Subgraphs<'a, Self::Subgraph> {
        self.schedule
            .batches
            .iter()
            .enumerate()
            .map(|(i, _)| i)
            .collect::<Vec<_>>()
            .into()
    }

    fn subgraph_nodes(&'a self, batch_index: &Self::Subgraph) -> dot2::Nodes<'a, Self::Node> {
        self.schedule.batches[*batch_index].as_slice().into()
    }
}
