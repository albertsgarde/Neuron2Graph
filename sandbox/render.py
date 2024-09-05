import graphviz  # type: ignore

path = "output/solu-2l/graphs/0_0.dot"

graphviz.render("dot", "pdf", path)  # type: ignore
