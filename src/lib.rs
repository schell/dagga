//! A crate for validating and scheduling directed acyclic graphs.
use rustc_hash::{FxHashMap, FxHashSet};
use snafu::prelude::*;

#[cfg(feature = "dot")]
pub mod dot;

/// An error in dag creation or scheduling.
#[derive(Debug, Snafu)]
pub enum DaggaError {
    #[snafu(display("Resources `{both_moved:?}` are moved both by {here} and {there}."))]
    MovedMoreThanOnce {
        both_moved: FxHashSet<usize>,
        here: String,
        there: String,
    },

    #[snafu(display("No root nodes"))]
    NoRootNodes,

    #[snafu(display("Missing node that results in resource {result}"))]
    MissingResult { result: usize },

    #[snafu(display("Cycle detected in graph of '{start}': {}", path.join(" -> ")))]
    Cycle { start: String, path: Vec<String> },

    #[snafu(display("{}", conflict_reason(&reqs)))]
    Conflict { reqs: Vec<Constraint> },

    #[snafu(display(
        "Cannot solve (at least) this constraint:\n  {constraint}\nPlease check that barriers are \
         not conflicting with other requirements"
    ))]
    CannotSolve { constraint: Constraint },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Op {
    Gt,
    Lt,
    Ne,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Constraint {
    lhs: String,
    rhs: String,
    op: Op,
    reasons: Vec<RequirementReason>,
}

impl std::fmt::Display for Constraint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&mk_req(self))
    }
}

impl Constraint {
    fn is_satisfied_by(&self, x: usize, y: usize) -> bool {
        match &self.op {
            Op::Gt => x > y,
            Op::Lt => x < y,
            Op::Ne => x != y,
        }
    }
}

/// Represents the search space of possible values for variables.
#[derive(Clone)]
struct Domain(Vec<usize>);

#[derive(Default)]
struct Solver {
    constraints: FxHashMap<String, Vec<Constraint>>,
    domains: FxHashMap<String, Domain>,
}

impl std::fmt::Display for Solver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Solver:\n")?;
        f.write_str("-constraints:\n")?;
        for (lhs, cs) in self.constraints.iter() {
            f.write_fmt(format_args!("--{}:\n", lhs))?;
            for (i, c) in cs.iter().enumerate() {
                f.write_fmt(format_args!("---{i}: {}\n", c))?;
            }
        }

        f.write_str("-domains:\n")?;
        for (lhs, domain) in self.domains.iter() {
            f.write_fmt(format_args!(
                "--{lhs}: {}\n",
                domain
                    .0
                    .iter()
                    .map(|r| format!("{:?}", r))
                    .collect::<Vec<_>>()
                    .join(",")
            ))?;
        }

        Ok(())
    }
}

/// Reduces the domains of variables using the AC-3 algo.
///
/// If any domains were reduced then that new set of reduced domains is
/// returned.
fn reduce_ac3(
    constraints: &FxHashMap<String, Vec<Constraint>>,
    domains: &FxHashMap<String, Domain>,
) -> Result<Option<FxHashMap<String, Domain>>, DaggaError> {
    let mut worklist: Vec<&Constraint> = constraints.values().flat_map(|cs| cs).collect();
    let mut domains = domains.clone();
    let mut domains_changed = false;
    let mut failure: Option<Constraint> = None;
    while let Some(constraint) = worklist.pop() {
        log::trace!("working on constraint: '{constraint}'");
        // arc-reduce
        let changed = {
            // UNWRAP: safe so long as Solver was constructed correctly
            let rhs_domain = domains.get(&constraint.rhs).unwrap().0.clone();
            let lhs_domain = domains.get_mut(&constraint.lhs).unwrap();
            let size_before = lhs_domain.0.len();
            lhs_domain.0.retain(|lhs_value| {
                let found = rhs_domain
                    .iter()
                    .any(|rhs_value| constraint.is_satisfied_by(*lhs_value, *rhs_value));
                if !found {
                    log::trace!(
                        "  removing {lhs_value} from the domain of {}",
                        constraint.lhs
                    );
                }
                found
            });
            let size_after = lhs_domain.0.len();
            size_before != size_after
        };
        if changed {
            domains_changed = true;
            log::trace!("  domain changed");
            // arc-reduce changed the domain...
            if domains.get(&constraint.lhs).unwrap().0.is_empty() {
                // ...but there are no viable values
                // left in the domain, which means we can't satisfy the constraint
                failure = Some(constraint.clone());
                break;
            } else {
                // ...and now we have to add any affected constraints back to the worklist
                // to continue solving
                let affected = constraints
                    .values()
                    .flat_map(|cs| cs)
                    .filter(|c| {
                        (c.lhs == constraint.lhs || c.rhs == constraint.lhs)
                            && c.rhs != constraint.rhs
                            && !worklist.contains(c)
                    })
                    .collect::<Vec<_>>();
                if !affected.is_empty() {
                    log::trace!("  adding back these constraints:");
                    for c in affected.iter() {
                        log::trace!("    {c}");
                    }
                    worklist.extend(affected);
                } else {
                    log::trace!("    but the worklist already contains all those affected");
                }
            }
        }
    }

    if let Some(constraint) = failure {
        return Err(DaggaError::CannotSolve { constraint });
    }

    if domains_changed {
        Ok(Some(domains))
    } else {
        Ok(None)
    }
}

impl Solver {
    fn new(dag: &Dag) -> Result<Self, DaggaError> {
        let mut solver = Solver::default();
        solver.constraints = dag.all_constraints()?;

        let size = dag.len();
        let domain = Domain((0..size).into_iter().collect());
        for node in dag.nodes() {
            solver.domains.insert(node.name(), domain.clone());
        }

        Ok(solver)
    }

    /// Reduces the domains of variables using the AC-3 algo until all domains
    /// are single valued.
    fn solve(&mut self) -> Result<(), DaggaError> {
        loop {
            if let Some(new_domains) = reduce_ac3(&self.constraints, &self.domains)? {
                self.domains = new_domains;
            }
            if let Some((_, domain)) = self.domains.iter_mut().find(|(_, d)| d.0.len() > 1) {
                // reduce the domain by hand and then continue
                domain.0.pop();
            } else {
                break;
            }
        }
        Ok(())
    }
}

fn mk_reason(reasons: &[RequirementReason]) -> String {
    reasons
        .iter()
        .map(|reason| match reason {
            RequirementReason::Barrier => "a barrier",
            RequirementReason::ExplicitOrder => "an explicit ordering (run_before or run_after)",
            RequirementReason::Input => "input requirements",
        })
        .collect::<Vec<_>>()
        .join(" and ")
}

fn mk_req(c: &Constraint) -> String {
    let a = c.lhs.as_str();
    let b = c.rhs.as_str();
    format!(
        "{a} should run {} {b} because of {}",
        match c.op {
            Op::Gt => "after",
            Op::Lt => "before",
            Op::Ne => "either before or after, but not in the same batch as",
        },
        mk_reason(&c.reasons)
    )
}

fn conflict_reason(reqs: &[Constraint]) -> String {
    format!(
        "Requirements are mutually exclusive:{}",
        reqs.into_iter()
            .map(|req| format!("- {}", mk_req(req)))
            .collect::<Vec<_>>()
            .join("\n")
    )
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RequirementReason {
    Barrier,
    ExplicitOrder,
    Input,
}

/// A named node in a graph.
#[derive(Debug, Default, Clone)]
pub struct Node {
    name: String,
    barrier: usize,
    moves: FxHashSet<usize>,
    reads: FxHashSet<usize>,
    writes: FxHashSet<usize>,
    results: FxHashSet<usize>,
    run_before: FxHashSet<String>,
    run_after: FxHashSet<String>,
}

impl Node {
    pub fn name(&self) -> String {
        self.name.clone()
    }

    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    pub fn with_move(mut self, rez: usize) -> Self {
        self.moves.insert(rez);
        self
    }

    pub fn with_moves(mut self, moves: impl Iterator<Item = usize>) -> Self {
        self.moves.extend(moves);
        self
    }

    pub fn with_read(mut self, rez: usize) -> Self {
        self.reads.insert(rez);
        self
    }

    pub fn with_reads(mut self, reads: impl IntoIterator<Item = usize>) -> Self {
        self.reads.extend(reads);
        self
    }

    pub fn with_write(mut self, rez: usize) -> Self {
        self.writes.insert(rez);
        self
    }

    pub fn with_writes(mut self, writes: impl IntoIterator<Item = usize>) -> Self {
        self.writes.extend(writes);
        self
    }

    pub fn with_result(mut self, rez: usize) -> Self {
        self.results.insert(rez);
        self
    }

    pub fn with_results(mut self, results: impl IntoIterator<Item = usize>) -> Self {
        self.results.extend(results);
        self
    }

    pub fn run_before(mut self, name: impl Into<String>) -> Self {
        self.run_before.insert(name.into());
        self
    }

    pub fn run_after(mut self, name: impl Into<String>) -> Self {
        self.run_after.insert(name.into());
        self
    }

    pub fn validate(&self) -> Result<(), DaggaError> {
        Ok(())
    }

    pub fn all_inputs(&self) -> FxHashSet<usize> {
        let mut all = self.moves.clone();
        all.extend(self.reads.clone());
        all.extend(self.writes.clone());
        all
    }

    /// Compare two nodes to determine the constraints between them.
    pub fn constraints(&self, other: &Node) -> Result<Vec<Constraint>, DaggaError> {
        let mut cs = FxHashMap::<Op, Vec<RequirementReason>>::default();
        if self.run_before.contains(&other.name) || other.run_after.contains(&self.name) {
            cs.insert(Op::Lt, vec![RequirementReason::ExplicitOrder]);
        } else if self.run_after.contains(&other.name) || other.run_before.contains(&self.name) {
            cs.insert(Op::Gt, vec![RequirementReason::ExplicitOrder]);
        }

        if self.barrier != other.barrier {
            let entry = cs
                .entry(
                    if self.barrier > other.barrier {
                        Op::Gt
                    } else {
                        Op::Lt
                    },
                )
                .or_default();
            entry.push(RequirementReason::Barrier);
        }

        let here_inputs = self.all_inputs();
        let there_inputs = other.all_inputs();

        let both_moved = self
            .moves
            .intersection(&other.moves)
            .copied()
            .collect::<FxHashSet<usize>>();
        snafu::ensure!(
            both_moved.len() == 0,
            MovedMoreThanOnceSnafu {
                both_moved,
                here: self.name(),
                there: other.name()
            }
        );

        // moves, then results
        let may_gt = if there_inputs.intersection(&self.moves).count() > 0 {
            // this node moves (consumes) a resource that the other requires
            Some(true)
        } else if here_inputs.intersection(&other.moves).count() > 0 {
            // this node requires a resource that the other moves (consumes)
            Some(false)
        } else if there_inputs.intersection(&self.results).count() > 0 {
            // this node results (creates) a resources that the other requires
            Some(false)
        } else if here_inputs.intersection(&other.results).count() > 0 {
            // thes node requires a resource that the other results in (creates)
            Some(true)
        } else {
            None
        };

        if let Some(gt) = may_gt {
            let entry = cs.entry(if gt { Op::Gt } else { Op::Lt }).or_default();
            entry.push(RequirementReason::Input);
        }

        // there is an exclusive borrow in the other of a resource this node requires
        // or
        // this node exclusively borrows a resource the other node requires
        if here_inputs.intersection(&other.writes).count() != 0
            || there_inputs.intersection(&self.writes).count() != 0
        {
            cs.insert(Op::Ne, vec![RequirementReason::Input]);
        }

        Ok(cs
            .into_iter()
            .map(|(op, reasons)| Constraint {
                lhs: self.name(),
                rhs: other.name(),
                op,
                reasons,
            })
            .collect())
    }
}

/// A directed acyclic graph.
#[derive(Debug, Default)]
pub struct Dag {
    barrier: usize,
    nodes: FxHashMap<String, Node>,
}

impl Dag {
    /// Returns the number of nodes in the DAG.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Add a node.
    pub fn with_node(mut self, node: Node) -> Self {
        self.add_node(node);
        self
    }

    /// Add a node.
    pub fn add_node(&mut self, mut node: Node) {
        node.barrier = self.barrier;
        self.nodes.insert(node.name(), node);
    }

    pub fn nodes(&self) -> impl Iterator<Item = &Node> {
        self.nodes.values()
    }

    /// Adds a barrier to the graph.
    ///
    /// A barrier will cause any nodes added after the barrier to be scheduled
    /// after the barrier.
    pub fn with_barrier(mut self) -> Self {
        self.barrier += 1;
        self
    }

    /// Generates a root node automatically, if need be.
    ///
    /// Searches the graph for node inputs that are missing (ie no node creates
    /// them as its result) and generates a node for them.
    ///
    /// This is useful
    pub fn generate_root_node(self) -> Self {
        let missing_inputs = self.get_missing_inputs();
        if !missing_inputs.is_empty() {
            self.with_node(
                Node::default()
                    .with_name("root")
                    .with_results(missing_inputs),
            )
        } else {
            self
        }
    }

    fn get_nodes_with_input(&self, result: usize) -> impl Iterator<Item = &Node> {
        self.nodes
            .values()
            .filter(move |node| node.all_inputs().contains(&result))
    }

    fn traverse_graph_from(&self, node: &Node, mut visited: Vec<String>) -> Result<(), DaggaError> {
        if visited.contains(&node.name) {
            snafu::ensure!(
                false,
                CycleSnafu {
                    start: node.name.clone(),
                    path: visited
                }
            );
        }
        visited.push(node.name.clone());
        for result in node.results.iter() {
            let result = *result;
            for next_node in self.get_nodes_with_input(result) {
                self.traverse_graph_from(next_node, visited.clone())?;
            }
        }
        Ok(())
    }

    pub fn root_nodes(&self) -> impl Iterator<Item = &Node> + '_ {
        self.nodes.values().filter(|node| {
            node.moves.is_empty()
                && node.reads.is_empty()
                && node.writes.is_empty()
                && node.run_after.len() == 0
        })
    }

    pub fn detect_cycles(&self) -> Result<(), DaggaError> {
        let mut has_root_nodes = false;
        for node in self.root_nodes() {
            has_root_nodes = true;
            self.traverse_graph_from(node, vec![])?;
        }
        snafu::ensure!(has_root_nodes, NoRootNodesSnafu);

        Ok(())
    }

    pub fn all_constraints(&self) -> Result<FxHashMap<String, Vec<Constraint>>, DaggaError> {
        let mut constraints: FxHashMap<String, Vec<Constraint>> = FxHashMap::default();
        for here in self.nodes.values() {
            let entry = constraints.entry(here.name()).or_default();
            for there in self.nodes.values() {
                if here.name() == there.name() {
                    continue;
                }
                let cs = here.constraints(there)?;
                entry.extend(cs);
            }
            entry.dedup_by(|a, b| a == b);
        }
        Ok(constraints)
    }

    /// Build the schedule from the current collection of nodes, if possible.
    pub fn build_schedule(&self) -> Result<Schedule, DaggaError> {
        self.detect_cycles()?;

        let mut solver = Solver::new(self)?;
        solver.solve()?;
        let mut batches: Vec<Vec<Node>> = vec![vec![]; self.nodes.len()];
        for (node_name, domain) in solver.domains.into_iter() {
            // UNWRAP: safe because these names came from the nodes themselves
            let node = self.nodes.get(&node_name).unwrap();
            let index = domain.0[0];
            batches[index].push(node.clone());
        }
        batches.retain(|batch| !batch.is_empty());
        Ok(Schedule { batches })
    }

    pub fn get_node_that_results_in(&self, result: usize) -> Option<&Node> {
        self.nodes
            .values()
            .find(|node| node.results.contains(&result))
    }

    /// Return any inputs that are missing from the graph.
    ///
    /// This function will return the inputs to nodes that are not created as
    /// the result of any node in the graph. These are inputs that would need
    /// to be created before the graph could be successfully run.
    pub fn get_missing_inputs(&self) -> FxHashSet<usize> {
        let mut all_inputs = FxHashSet::default();
        let mut all_results = FxHashSet::default();
        for node in self.nodes.values() {
            all_inputs.extend(node.all_inputs());
            all_results.extend(node.results.clone());
        }

        all_inputs.difference(&all_results).copied().collect()
    }

    pub fn get_node(&self, name: impl AsRef<str>) -> Option<&Node> {
        self.nodes.get(name.as_ref())
    }
}

/// A built dag schedule.
#[derive(Debug)]
pub struct Schedule {
    pub batches: Vec<Vec<Node>>,
}

impl Schedule {
    pub fn batched_names(&self) -> Vec<Vec<&str>> {
        self.batches
            .iter()
            .map(|batch| batch.iter().map(|node| node.name.as_str()).collect())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use crate::dot::{save_as_dot, DagLegend};

    use super::*;

    fn dag_schedule(dag: &Dag) -> Vec<String> {
        let schedule = dag.build_schedule().unwrap();
        schedule
            .batched_names()
            .iter()
            .map(|names| names.join(", "))
            .collect::<Vec<_>>()
    }

    fn as_strs(vs: &Vec<String>) -> Vec<&str> {
        vs.iter().map(|s| s.as_str()).collect::<Vec<_>>()
    }

    fn assert_batches(expected: &[&str], dag: &Dag) {
        let batches = dag_schedule(&dag);
        assert_eq!(expected, as_strs(&batches).as_slice());
    }

    #[test]
    fn sanity() {
        // Create names for our resources.
        //
        // These represent the types of the resources that get created, passed through
        // and consumed by each node.
        let [a, b, c, d]: [usize; 4] = [0, 1, 2, 3];

        // This node results in the creation of an `a`.
        let create_a = Node::default().with_name("create-a").with_result(a);
        // This node creates `b`
        let create_b = Node::default().with_name("create-b").with_result(b);
        // This node reads `a` and `b` and results in `c`
        let create_c = Node::default()
            .with_name("create-c")
            .with_read(a)
            .with_read(b)
            .with_result(c);
        // This node modifies `a`, but for reasons outside of the scope of the types
        // expressed here (just as an example), it must be run before
        // "create-c". There is no result of this node beside the side-effect of
        // modifying `a`.
        let modify_a = Node::default()
            .with_name("modify-a")
            .with_write(a)
            .with_read(b)
            .run_before("create-c");
        assert!(modify_a.run_before.contains("create-c"));
        // This node consumes `a`, `b`, `c` and results in `d`.
        let reduce_abc_to_d = Node::default()
            .with_name("reduce-abc-to-d")
            .with_move(a)
            .with_move(b)
            .with_move(c)
            .with_result(d);

        // Add the nodes with their dependencies and build the schedule.
        // The order they are added should not matter (it may cause differences in
        // scheduling, but always result in a valid schedule).
        let mut dag = Dag::default();

        dag.add_node(create_a);
        assert_batches(&["create-a"], &dag);

        dag.add_node(create_b);
        assert_batches(&["create-a, create-b"], &dag);

        dag.add_node(create_c);
        assert_batches(&["create-a, create-b", "create-c"], &dag);
        let dag = dag.with_node(reduce_abc_to_d).with_node(modify_a);

        assert_batches(
            &[
                "create-a, create-b", /* each batch can be run in parallel w/o violating
                                       * exclusive borrows */
                "modify-a",
                "create-c",
                "reduce-abc-to-d",
            ],
            &dag,
        );

        let legend = DagLegend::new(dag)
            .with_name("example")
            .with_resource("A", a)
            .with_resource("B", b)
            .with_resource("C", c);
        save_as_dot(&legend, "example.dot").unwrap();
    }

    #[test]
    #[should_panic]
    fn detect_cycle() {
        let [a, b, c] = [0, 1, 2usize];
        let schedule = Dag::default()
            .with_node(Node::default().with_name("a").with_result(a))
            .with_node(
                Node::default()
                    .with_name("b")
                    .with_read(a)
                    .with_read(c)
                    .with_result(b),
            )
            .with_node(Node::default().with_name("c").with_read(b).with_result(c))
            .build_schedule()
            .unwrap();
        println!("{:?}", schedule.batched_names());
    }

    #[test]
    #[should_panic]
    fn detect_unsolvable_barrier() {
        let dag = Dag::default()
            .with_node(Node::default().with_name("create-0").with_result(0))
            .with_node(Node::default().with_name("read-1").with_read(1))
            .with_barrier()
            .with_node(Node::default().with_name("create-1").with_result(1));
        assert_batches(&["blah"], &dag);
    }

    #[test]
    fn without_results() {
        let [a, b, c] = [0, 1, 2usize];
        let dag = Dag::default()
            .with_node(Node::default().with_name("run").with_read(a))
            .with_node(Node::default().with_name("jog").with_write(b).with_move(c));
        assert_eq!(
            vec![a, b, c],
            dag.get_missing_inputs().into_iter().collect::<Vec<_>>()
        );

        let dag = dag.generate_root_node();
        let batches = dag_schedule(&dag);
        assert_eq!(["root", "jog, run"], as_strs(&batches).as_slice());
    }
}
