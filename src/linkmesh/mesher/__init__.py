"""
mesher - LinkMesh OFT/TokaMaker-Triangle mesh generation package.

Official interface:
    linkmesh.mesher.oft_adapter.generate_mesh(gfile_data, ...) -> Mesh2D

Official data flow:
    GFileData
    -> build_oft_domain_from_gfile()
    -> OFT gs_Domain
    -> mesh_from_oft_domain()
    -> Mesh2D

Constraints:
    - OpenFUSIONToolkit 26.6 is the only LinkMesh mesh-generation backend.
    - Triangle is called internally by OFT through liboft_triangle; LinkMesh
      does not directly depend on the Python triangle package.
    - Do not introduce a second mesh-generation backend.
    - Do not introduce three-dimensional mesh-generation logic.
    - Mesh2D is defined only in linkmesh.mesh.
"""