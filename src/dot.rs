//! Support for serializing [`Dag`](super::Dag) as a dot file, for use with graphiz.
use std::collections::HashMap;

use super::*;

#[derive(Debug, Snafu)]
pub enum DotError {
    #[snafu(display("Could not create file: {}", source))]
    CreateFile { source: std::io::Error },

    #[snafu(display("{}", source))]
    Dot { source: dot2::Error },
}

/// A `Dag` and some labels for providing user-facing names for resources.
pub struct DagLegend {
    pub resources: HashMap<usize, String>,
    pub name: String,
    pub dag: Dag,
}

impl DagLegend {
    pub fn new(dag: Dag) -> Self {
        DagLegend { resources: Default::default(), name: String::new(), dag }
    }

    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    pub fn with_resource(mut self, name: impl Into<String>, resource: usize) -> Self {
        self.resources.insert(resource, name.into());
        self
    }
}

pub fn save_as_dot(legend: &DagLegend, path: impl AsRef<std::path::Path>) -> Result<(), DotError> {
    let mut file = std::fs::File::create(path).context(CreateFileSnafu)?;
    dot2::render(&legend, &mut file).context(DotSnafu)
}

#[derive(Clone)]
pub struct Edge {
    rez: usize,
    node: String,
}

impl<'a> dot2::Labeller<'a> for &'a DagLegend {
    type Node = Node;

    type Edge = Edge;

    type Subgraph = usize;

    fn graph_id(&'a self) -> dot2::Result<dot2::Id<'a>> {
        dot2::Id::new(&self.name)
    }

    fn node_id(&'a self, n: &Self::Node) -> dot2::Result<dot2::Id<'a>> {
        dot2::Id::new(n.name.replace("-", "_").to_string())
    }

    fn edge_label(&'a self, e: &Self::Edge) -> dot2::label::Text<'a> {
        dot2::label::Text::LabelStr(
            self.resources
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
}

fn get_edges(dag: &Dag, node: &Node) -> Vec<Edge> {
    let mut edges = vec![];
    for result in node.results.iter() {
        for downstream_node in dag.get_nodes_with_input(*result) {
            edges.push(Edge {
                rez: *result,
                node: downstream_node.name.clone(),
            });
        }
    }
    edges
}

impl<'a> dot2::GraphWalk<'a> for &'a DagLegend {
    type Node = Node;

    type Edge = Edge;

    type Subgraph = usize;

    fn nodes(&'a self) -> dot2::Nodes<'a, Self::Node> {
        let batches: Vec<Node> = self.dag.nodes.values().cloned().collect();
        batches.into()
    }

    fn edges(&'a self) -> dot2::Edges<'a, Self::Edge> {
        let mut edges: Vec<Edge> = vec![];
        for node in self.dag.nodes.values() {
            edges.extend(get_edges(&self.dag, node));
        }
        edges.into()
    }

    fn source(&'a self, edge: &Self::Edge) -> Self::Node {
        self.dag.get_node_that_results_in(edge.rez).unwrap().clone()
    }

    fn target(&'a self, edge: &Self::Edge) -> Self::Node {
        self.dag.nodes.get(&edge.node).unwrap().clone()
    }

    fn subgraphs(&'a self) -> dot2::Subgraphs<'a, Self::Subgraph> {
        let schedule = self.dag.build_schedule().unwrap();
        schedule
            .batches
            .iter()
            .enumerate()
            .map(|(i, _)| i)
            .collect::<Vec<_>>()
            .into()
    }

    fn subgraph_nodes(&'a self, batch_index: &Self::Subgraph) -> dot2::Nodes<'a, Self::Node> {
        let schedule = self.dag.build_schedule().unwrap();
        schedule.batches.get(*batch_index).unwrap().clone().into()
    }
}
