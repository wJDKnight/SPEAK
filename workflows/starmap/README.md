# STARmap workflow

Scope: BZ5, BZ9, and BZ14.  Their `data.csv`, `celltype.csv`, `pos.csv`, and
`domain.csv` files are bundled in `examples/starmap/<sample>/`.  The exact gene
list used by the method is `examples/starmap/representative_genes.txt`.

The principal reproducibility path is preprocessing/markers, batch generation,
model response retrieval, spatial refinement, and metric recomputation.  The
older one-shot notebooks are retained for provenance but are not the canonical
paper entry point.
