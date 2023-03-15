# dagga 🌿
A crate for scheduling directed acyclic graphs.

## Features

- node `creates` resources semantics
- node `reads` resource semantics, ie borrow
- `writes` resource semantics, ie mutable/exclusive borrow
- `consumes` resource semantics, ie move
- node dependencies
  * node X must run _before_ node Y
  * node X must run _after_ node Y
  * barriers - nodes added before a barrier will always be scheduled before the barrier and nodes added after a barrier will always be scheduled after the barrier

## Example uses
* scheduling parallel operations with dependencies, shared and exclusive resources
* scheduling steps in a render graph
* scheduling system batches in ECS
* scheduling audio nodes in an audio graph

## Example

```rust
use dagga::*;

// Create names/values for our resources.
//
// These represent the types of the resources that get created, passed through
// and consumed by each node.
let [a, b, c, d]: [usize; 4] = [0, 1, 2, 3];

// Add the nodes with their dependencies and build the schedule.
// The order they are added should not matter (it may cause differences in
// scheduling, but always result in a valid schedule).
let dag = Dag::<(), usize>::default()
    .with_node({
        // This node results in the creation of an `a`.
        Node::new(()).with_name("create-a").with_result(a)
    })
    .with_node({
        // This node creates a `b`.
        Node::new(()).with_name("create-b").with_result(b)
    })
    .with_node({
        // This node reads `a` and `b` and results in `c`
        Node::new(())
            .with_name("create-c")
            .with_read(a)
            .with_read(b)
            .with_result(c)
    })
    .with_node({
        // This node modifies `a`, but for reasons outside of the scope of the types
        // expressed here (just as an example), it must be run before
        // "create-c". There is no result of this node beside the side-effect of
        // modifying `a`.
        Node::new(())
            .with_name("modify-a")
            .with_write(a)
            .with_read(b)
            .run_before("create-c")
    })
    .with_node({
        // This node consumes `a`, `b`, `c` and results in `d`.
        Node::new(())
            .with_name("reduce-abc-to-d")
            .with_move(a)
            .with_move(b)
            .with_move(c)
            .with_result(d)
    });

dagga::assert_batches(
    &[
        "create-a, create-b", /* each batch can be run in parallel w/o violating
                                * exclusive borrows */
        "modify-a",
        "create-c",
        "reduce-abc-to-d",
    ],
    dag.clone(),
);
```

You can also have `dagga` create a dot graph file to visualize the schedule (using graphiz or similar):
![dagga example schedule](example.svg)
