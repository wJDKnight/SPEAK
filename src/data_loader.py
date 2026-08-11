import os
import pandas as pd
import scanpy as sc
import anndata as ad
from typing import Optional, List, Dict, Union

def load_spatial_data_anndata(
    data_path: str,
    adata_file: str,
    config: object = None,
    celltype_path: Optional[str] = "celltype.csv",
    rename_celltype: bool = False,  # if True, use config.celltype_rename to rename the celltype
    domain_file: Optional[str] = "domain.csv",
    optional_files: Optional[List[str]] = None,
    index_col: int = 0
) -> ad.AnnData:
    """
    Load spatial transcriptomics data from an AnnData H5AD file with optional metadata.

    This function loads an AnnData object and optionally joins it with cell type,
    domain, and other metadata files. It handles both single-column and multi-column
    cell type data, automatically creating one-hot encodings when needed.

    Parameters:
    -----------
    data_path : str
        Path to the directory containing the H5AD file and metadata files
    adata_file : str
        Name of the H5AD file containing the main AnnData object
    config : object, optional
        Configuration object containing attribute names and settings.
        Expected attributes: name_truth, celltype_name, celltype_rename
    celltype_path : str, optional
        Name of the cell type annotation file (default: "celltype.csv")
    rename_celltype : bool, optional
        If True, use config.celltype_rename to rename cell types (default: False)
    domain_file : str, optional
        Name of the domain/truth labels file (default: "domain.csv")
    optional_files : List[str], optional
        List of additional CSV files to load and join to observations
    index_col : int, optional
        Column to use as row index for metadata files (default: 0)

    Returns:
    --------
    adata : anndata.AnnData
        AnnData object with expression data and joined metadata

    Raises:
    -------
    FileNotFoundError
        If the main H5AD file doesn't exist
    KeyError
        If required spatial coordinates or config attributes are missing
    ValueError
        If cell type column is not found and no celltype file exists
    """

    # Validate inputs
    adata_path = os.path.join(data_path, adata_file)
    if not os.path.exists(adata_path):
        raise FileNotFoundError(f"AnnData file not found: {adata_path}")

    # check if the data is Visium
    if adata_file == "filtered_feature_bc_matrix.h5":
        print("Reading Visium data")
        adata = sc.read_visium(data_path)
        adata.var_names_make_unique()
    else:
        print("Reading AnnData from h5ad file")
        adata = ad.read_h5ad(adata_path)

    # Filter data based on truth labels if config is provided
    if config is not None and hasattr(config, 'name_truth') and config.name_truth:
        if config.name_truth in adata.obs.columns:
            initial_size = adata.n_obs
            # Remove rows with NaN values in the truth column
            adata = adata[~adata.obs[config.name_truth].isna()].copy()
            filtered_size = adata.n_obs
            print(f"Filtered data: {initial_size} -> {filtered_size} cells "
                  f"(removed {initial_size - filtered_size} cells with NaN in '{config.name_truth}')")
        else:
            print(f"Warning: Truth column '{config.name_truth}' not found in adata.obs")

    # Extract and add spatial coordinates to observations
    if 'spatial' not in adata.obsm:
        raise KeyError("Spatial coordinates not found in adata.obsm['spatial']")

    try:
        pos_data = pd.DataFrame(adata.obsm['spatial'], columns=['x', 'y'], index=adata.obs_names)
        adata.obs = adata.obs.join(pos_data)
        print(f"Added spatial coordinates (x, y) to observations")
    except Exception as e:
        raise ValueError(f"Error processing spatial coordinates: {e}")
    # Initialize list to store metadata DataFrames
    metadata_dfs = []

    # Load cell type data from external file or existing observations
    resolved_celltype_path = None
    if celltype_path:
        resolved_celltype_path = (
            celltype_path
            if os.path.isabs(celltype_path)
            else os.path.join(data_path, celltype_path)
        )
    if resolved_celltype_path and os.path.exists(resolved_celltype_path):
        print(f"Loading cell type data from: {resolved_celltype_path}")
        try:
            celltype_data = pd.read_csv(resolved_celltype_path, index_col=index_col)

            # Validate that config has required attributes for cell type processing
            if config is None or not hasattr(config, 'celltype_name'):
                raise ValueError("Config object with 'celltype_name' attribute required for cell type processing")

            # Handle single-column cell type data (categorical labels)
            if celltype_data.shape[1] == 1:
                column_name = config.celltype_name
                celltype_data.columns = [column_name]
                print(f"Converting single celltype column '{column_name}' to one-hot encoding")

                # Create one-hot encoding for cell type proportions
                cellprop_data = pd.get_dummies(celltype_data[column_name])
                print(f"Created one-hot encoding with {cellprop_data.shape[1]} cell types")

            # Handle multi-column cell type data (proportion data)
            else:
                print(f"Multi-column celltype data detected with {celltype_data.shape[1]} columns. "
                      f"Finding dominant cell type for each cell.")

                # Store the original proportion data
                cellprop_data = celltype_data.copy()

                # Get the column name with the maximum value for each row
                dominant_celltype = celltype_data.idxmax(axis=1)

                # Create new single-column dataframe with dominant cell types
                column_name = config.celltype_name
                celltype_data = pd.DataFrame({column_name: dominant_celltype}, index=celltype_data.index)
                print(f"Created single celltype column '{column_name}' with dominant cell types")

            # Ensure indices match with adata observations
            celltype_data = celltype_data.reindex(adata.obs_names, fill_value=None)
            cellprop_data = cellprop_data.reindex(adata.obs_names, fill_value=0.0)

            metadata_dfs.append(celltype_data)
            metadata_dfs.append(cellprop_data)

        except Exception as e:
            print(f"Error loading cell type data: {e}")
            raise

    else:
        print(f"Cell type file not found: {resolved_celltype_path or 'None'}")
        print("Attempting to find cell type information in adata.obs...")

        # Check if config and celltype_name are available
        if config is None or not hasattr(config, 'celltype_name'):
            print("Error: No config.celltype_name provided and no cell type file found")
            raise ValueError("No config.celltype_name provided and no cell type file found")
        else:
            # Check if the celltype_name column exists in adata.obs
            if config.celltype_name in adata.obs.columns:
                print(f"Found cell type column '{config.celltype_name}' in adata.obs")

                # Apply cell type renaming if requested
                if rename_celltype and hasattr(config, 'celltype_rename') and config.celltype_rename:
                    print(f"Renaming cell types using config.celltype_rename")
                    adata.obs[config.celltype_name] = adata.obs[config.celltype_name].map(config.celltype_rename)

                # Create one-hot encoding from existing cell type column
                cellprop_data = pd.get_dummies(adata.obs[config.celltype_name])
                print(f"Created one-hot encoding with {cellprop_data.shape[1]} cell types from existing data")
                metadata_dfs.append(cellprop_data)
            else:
                print(f"Available columns: {list(adata.obs.columns)}")
                raise ValueError(f"Cell type column '{config.celltype_name}' not found in adata.obs")

    # Load domain/truth data (optional)
    if domain_file:
        domain_path = os.path.join(data_path, domain_file)
        if os.path.exists(domain_path):
            print(f"Loading domain data from: {domain_path}")
            try:
                domain_data = pd.read_csv(domain_path, index_col=index_col)

                # Rename domain column if config specifies a truth name
                if config is not None and hasattr(config, 'name_truth') and config.name_truth:
                    if domain_data.shape[1] == 1:
                        domain_data.columns = [config.name_truth]
                        print(f"Renamed domain column to '{config.name_truth}'")
                    else:
                        raise ValueError(f"Domain file has {domain_data.shape[1]} columns, "
                                         f"expected 1 for renaming to '{config.name_truth}'")

                # Ensure indices match with adata observations
                domain_data = domain_data.reindex(adata.obs_names, fill_value=None)
                metadata_dfs.append(domain_data)

            except Exception as e:
                print(f"Error loading domain data: {e}")
                raise
        else:
            print(f"Info: Domain file not found (optional): {domain_path}")

    # Load additional optional files
    if optional_files:
        print(f"Loading {len(optional_files)} optional files...")
        for i, optional_file in enumerate(optional_files):
            optional_path = os.path.join(data_path, optional_file)
            if os.path.exists(optional_path):
                print(f"Loading optional data from: {optional_path}")
                try:
                    # read csv or tsv
                    if optional_file.endswith(".csv"):
                        optional_data = pd.read_csv(optional_path, index_col=index_col)
                    elif optional_file.endswith(".tsv"):
                        optional_data = pd.read_csv(optional_path, index_col=index_col, sep="\t")
                    else:
                        raise ValueError(f"Unsupported file type: {optional_file}")

                    # Ensure indices match with adata observations
                    optional_data = optional_data.reindex(adata.obs_names, fill_value=None)
                    metadata_dfs.append(optional_data)

                except Exception as e:
                    print(f"Error loading optional file {optional_file}: {e}")
                    # Continue with other files instead of failing completely
                    continue
            else:
                print(f"Warning: Optional file not found: {optional_path}")

    # Join all metadata to adata.obs with error handling
    if metadata_dfs:
        print(f"Joining {len(metadata_dfs)} metadata DataFrames to observations...")
        try:
            # Filter out None values and empty DataFrames
            valid_metadata_dfs = [df for df in metadata_dfs
                                if df is not None and not df.empty]

            if valid_metadata_dfs:
                # Join all metadata at once
                for i, df in enumerate(valid_metadata_dfs):
                    try:
                        print(f"Joining metadata DataFrames")
                        initial_rows = adata.n_obs
                        adata.obs = adata.obs.join(df, how='left')

                        # Check if the joined DataFrame contains target columns and remove NaN rows
                        target_columns = []
                        if config is not None:
                            if hasattr(config, 'celltype_name') and config.celltype_name and config.celltype_name in df.columns:
                                target_columns.append(config.celltype_name)
                            if hasattr(config, 'name_truth') and config.name_truth and config.name_truth in df.columns:
                                target_columns.append(config.name_truth)

                        if target_columns:
                            # Remove rows with NaN values in any of the target columns
                            for col in target_columns:
                                if col in adata.obs.columns:
                                    nan_mask = adata.obs[col].isna()
                                    if nan_mask.any():
                                        adata = adata[~nan_mask].copy()
                                        rows_removed = initial_rows - adata.n_obs
                                        print(f"Info: Removed {rows_removed} rows with NaN values in column '{col}' "
                                              f"({initial_rows} -> {adata.n_obs} cells)")
                                        initial_rows = adata.n_obs  # Update for next column

                        print(f"Successfully joined metadata DataFrame {i+1}/{len(valid_metadata_dfs)}")
                    except Exception as e:
                        print(f"Warning: Failed to join metadata DataFrame {i+1}: {e}")
                        continue
            else:
                print("No valid metadata DataFrames to join")

        except Exception as e:
            print(f"Error during metadata joining: {e}")
            raise
    else:
        print("No additional metadata to join")

    # Final data validation and summary
    print("\n" + "="*60)
    print("DATA LOADING SUMMARY")
    print("="*60)
    print(f"✓ Data loading complete. Shape: {adata.shape}")
    print(f"✓ Observations (cells): {adata.n_obs:,}")
    print(f"✓ Variables (genes): {adata.n_vars:,}")
    print(f"✓ Metadata columns: {len(adata.obs.columns)}")
    print(f"  Columns: {list(adata.obs.columns)}")

    # Check for any missing data
    missing_coords = adata.obs[['x', 'y']].isna().any().any()
    if missing_coords:
        print(f"⚠️  Warning: Some spatial coordinates are missing")

    print("="*60)

    return adata

def load_spatial_data_csv(
    data_path: str,
    main_data_file: str = "data.csv",
    celltype_file: str = "celltype.csv",
    pos_file: str = "pos.csv",
    domain_file: Optional[str] = "domain.csv",
    config: object = None,
    index_col: int = 0,
    first_column_names: bool = True,
    optional_files: Optional[List[str]] = None

) -> ad.AnnData:
    """
    Load spatial transcriptomics data from a folder containing CSV files.

    Parameters:
    -----------
    data_path : str
        Path to the directory containing CSV files
    main_data_file : str, default "data.csv"
        Name of the main expression data file
    celltype_file : str, default "celltype.csv"
        Name of the cell type annotation file
    pos_file : str, default "pos.csv"
        Name of the position/coordinates file
    domain_file : str, default "domain.csv"
        Name of the domain/truth labels file (optional)
    index_col : int, default 0
        Column to use as row index for metadata files
    first_column_names : bool, default True
        Whether first column contains gene names in main data file
    optional_files : List[str], optional
        List of additional CSV files to load and join
    config : Optional[object], optional
        Additional configuration parameters

    Returns:
    --------
    adata : anndata.AnnData
        AnnData object with expression data and metadata

    Examples:
    ---------
    >>> # Basic usage
    >>> adata = load_spatial_data_csv("/path/to/data/")

    >>> # With custom file names and truth column
    >>> adata = load_spatial_data_csv(
    ...     "/path/to/data/",
    ...     main_data_file="expression.csv"
    ... )

    >>> # With additional metadata files
    >>> adata = load_spatial_data_csv(
    ...     "/path/to/data/",
    ...     optional_files=["batch_info.csv", "quality_metrics.csv"]
    ... )
    """

    # Validate data path
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data path does not exist: {data_path}")

    # Load main expression data
    main_file_path = os.path.join(data_path, main_data_file)
    if not os.path.exists(main_file_path):
        raise FileNotFoundError(f"Main data file not found: {main_file_path}")

    print(f"Loading main data from: {main_file_path}")
    adata = sc.read_csv(main_file_path, first_column_names=first_column_names)

    # Add cell ID to adata.obs
    adata.obs['cell_id'] = adata.obs_names

    # Initialize list to store metadata DataFrames
    metadata_dfs = []

    # Load cell type data
    celltype_path = os.path.join(data_path, celltype_file)
    if os.path.exists(celltype_path):
        print(f"Loading cell type data from: {celltype_path}")
        celltype_data = pd.read_csv(celltype_path, index_col=index_col)

        # If celltype_data has only one column, create a one-hot encoding dataframe
        if celltype_data.shape[1] == 1:
            column_name = config.celltype_name
            celltype_data.columns = [column_name]
            print(f"Converting single celltype column '{column_name}' to one-hot encoding")
            cellprop_data = pd.get_dummies(celltype_data[column_name])
        else:
            # If multiple columns, find cell type with largest value for each row
            print(f"Multi-column celltype data detected. Finding dominant cell type for each cell.")
            cellprop_data = celltype_data
            # Get the column name with the maximum value for each row
            dominant_celltype = celltype_data.idxmax(axis=1)

            # Create new single-column dataframe
            column_name = config.celltype_name
            celltype_data = pd.DataFrame({column_name: dominant_celltype}, index=celltype_data.index)
            print(f"Created single celltype column '{column_name}' with dominant cell types")

        metadata_dfs.append(celltype_data)
        metadata_dfs.append(cellprop_data)
    else:
        print(f"Warning: Cell type file not found: {celltype_path}")

    # Load position data
    pos_path = os.path.join(data_path, pos_file)
    if os.path.exists(pos_path):
        print(f"Loading position data from: {pos_path}")
        pos_data = pd.read_csv(pos_path, index_col=index_col)
        metadata_dfs.append(pos_data)
    else:
        print(f"Warning: Position file not found: {pos_path}")

    # Load domain/truth data (optional)
    domain_path = os.path.join(data_path, domain_file) if domain_file else None
    if domain_path and os.path.exists(domain_path):
        print(f"Loading domain data from: {domain_path}")
        domain_data = pd.read_csv(domain_path, index_col=index_col)
        if config.name_truth:
            domain_data.columns = [config.name_truth]
        metadata_dfs.append(domain_data)
    else:
        print(f"Info: Domain file not found (optional): {domain_path or 'not configured'}")

    # Load additional optional files
    if optional_files:
        for optional_file in optional_files:
            optional_path = os.path.join(data_path, optional_file)
            if os.path.exists(optional_path):
                print(f"Loading optional data from: {optional_path}")
                optional_data = pd.read_csv(optional_path, index_col=index_col)
                metadata_dfs.append(optional_data)
            else:
                print(f"Warning: Optional file not found: {optional_path}")

    # Join all metadata to adata.obs
    if metadata_dfs:
        # Remove None values and join
        valid_metadata_dfs = [df for df in metadata_dfs if df is not None]
        if valid_metadata_dfs:
            initial_rows = adata.n_obs
            adata.obs = adata.obs.join(valid_metadata_dfs)

            # Check if the joined DataFrame contains target columns and remove NaN rows
            target_columns = []
            if config is not None:
                if hasattr(config, 'celltype_name') and config.celltype_name and config.celltype_name in adata.obs.columns:
                    target_columns.append(config.celltype_name)
                if hasattr(config, 'name_truth') and config.name_truth and config.name_truth in adata.obs.columns:
                    target_columns.append(config.name_truth)

            if target_columns:
                # Remove rows with NaN values in any of the target columns
                for col in target_columns:
                    nan_mask = adata.obs[col].isna()
                    if nan_mask.any():
                        adata = adata[~nan_mask].copy()
                        rows_removed = initial_rows - adata.n_obs
                        print(f"Removed {rows_removed} rows with NaN values in column '{col}' "
                              f"({initial_rows} -> {adata.n_obs} cells)")
                        initial_rows = adata.n_obs  # Update for next column



    print(f"Data loading complete. Shape: {adata.shape}")
    print(f"Observations (cells): {adata.n_obs}")
    print(f"Variables (genes): {adata.n_vars}")
    print(f"Metadata columns: {list(adata.obs.columns)}")

    return adata


def load_spatial_data_onefile(
    data_file_name: str,
    config: object,
    index_col: int = 0,
    representetive_gene_list: Optional[List[str]] = None
) -> ad.AnnData:
    """
    celltype and position are in one file. sometimes there are some genes in the file as well.
    """
    all_in_one_df = pd.read_csv(os.path.join(data_file_name), index_col=index_col)

    if representetive_gene_list is None:
        adata = ad.AnnData(X=None, obs=all_in_one_df)
    else:
        genes_data = all_in_one_df.loc[:, representetive_gene_list]
        other_data = all_in_one_df.loc[:, ~all_in_one_df.columns.isin(representetive_gene_list)]
        adata = ad.AnnData(X=genes_data, obs=other_data)

    # creat one-hot encoding for cell type
    celltype_data = pd.get_dummies(adata.obs[config.celltype_name])
    adata.obs = adata.obs.join(celltype_data)

    return adata


def get_data_info(data_path: str) -> Dict[str, bool]:
    """
    Check what CSV files are available in the data directory.

    Parameters:
    -----------
    data_path : str
        Path to the data directory

    Returns:
    --------
    file_info : Dict[str, bool]
        Dictionary showing which standard files are present
    """

    standard_files = {
        'data.csv': 'main_data',
        'celltype.csv': 'celltype',
        'pos.csv': 'position',
        'domain.csv': 'domain'
    }

    file_info = {}

    print(f"Checking files in: {data_path}")
    print("-" * 50)

    for filename, description in standard_files.items():
        file_path = os.path.join(data_path, filename)
        exists = os.path.exists(file_path)
        file_info[description] = exists
        status = "✓" if exists else "✗"
        print(f"{status} {filename} ({description})")

    # List any other CSV files
    try:
        all_files = os.listdir(data_path)
        csv_files = [f for f in all_files if f.endswith('.csv')]
        other_csvs = [f for f in csv_files if f not in standard_files.keys()]

        if other_csvs:
            print("\nOther CSV files found:")
            for f in other_csvs:
                print(f"  • {f}")
    except OSError:
        print(f"Error: Cannot access directory {data_path}")

    return file_info
