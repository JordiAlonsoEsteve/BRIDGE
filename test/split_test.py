import numpy as np
import pandas as pd
import pytest
from bridge.split.split import structured_cv_split, structured_train_test_split


@pytest.fixture
def mock_genomic_data():
    """Generates synthetic X, C, Y DataFrames and row groups for testing."""
    n_genes = 10
    n_tfs = 5
    n_samples = 20
    n_covariates = 3

    genes = [f"gene_{i}" for i in range(n_genes)]
    tfs = [f"tf_{i}" for i in range(n_tfs)]
    samples = [f"sample_{i}" for i in range(n_samples)]

    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(n_tfs, n_samples), index=tfs, columns=samples)
    C = pd.DataFrame(
        np.random.randn(n_genes, n_covariates),
        index=genes,
        columns=[f"cov_{i}" for i in range(n_covariates)],
    )
    Y = pd.DataFrame(np.random.randn(n_genes, n_samples), index=genes, columns=samples)

    # 5 distinct groups across 10 genes
    row_groups = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])

    return X, C, Y, row_groups


# ---------------------------------------------------------------------------
# Tests for structured_train_test_split
# ---------------------------------------------------------------------------
class TestStructuredTrainTestSplit:

    def test_standard_split_shapes_and_types(self, mock_genomic_data):
        """Verify train/test dataset dimensions and return types without row groups."""
        X, C, Y, _ = mock_genomic_data
        test_size = 0.2

        (
            X_train,
            X_test,
            C_train,
            C_test,
            Y_train,
            Y_test,
            rg_train,
            rg_test,
        ) = structured_train_test_split(X, C, Y, test_size=test_size, random_state=42)

        # 10 genes: 8 train, 2 test | 20 samples: 16 train, 4 test
        assert Y_train.shape == (8, 16)
        assert Y_test.shape == (2, 4)
        assert X_train.shape == (5, 16)
        assert X_test.shape == (5, 4)
        assert C_train.shape == (8, 3)
        assert C_test.shape == (2, 3)

        assert rg_train is None
        assert rg_test is None

    def test_grouped_split_no_group_leakage(self, mock_genomic_data):
        """Ensure row groups are strictly isolated between train and test sets."""
        X, C, Y, row_groups = mock_genomic_data

        (
            _,
            _,
            _,
            _,
            _,
            _,
            rg_train,
            rg_test,
        ) = structured_train_test_split(
            X, C, Y, row_groups=row_groups, test_size=0.2, random_state=42
        )

        assert rg_train is not None
        assert rg_test is not None

        # Train and test groups must be completely disjoint
        train_groups = set(rg_train)
        test_groups = set(rg_test)
        assert train_groups.isdisjoint(test_groups)

    def test_index_and_column_alignment(self, mock_genomic_data):
        """Verify that sliced DataFrames maintain corresponding index/column labels."""
        X, C, Y, _ = mock_genomic_data

        X_tr, X_te, C_tr, C_te, Y_tr, Y_te, _, _ = structured_train_test_split(
            X, C, Y, random_state=42
        )

        # Y rows match C rows; Y columns match X columns
        assert Y_tr.index.equals(C_tr.index)
        assert Y_te.index.equals(C_te.index)
        assert Y_tr.columns.equals(X_tr.columns)
        assert Y_te.columns.equals(X_te.columns)

    def test_assertion_mismatched_c_y_index(self, mock_genomic_data):
        """Should raise AssertionError if C and Y have different gene indices."""
        X, C, Y, _ = mock_genomic_data
        C_bad = C.copy()
        C_bad.index = [f"wrong_gene_{i}" for i in range(len(C))]

        with pytest.raises(AssertionError, match="same row index as Y"):
            structured_train_test_split(X, C_bad, Y)

    def test_assertion_mismatched_x_y_columns(self, mock_genomic_data):
        """Should raise AssertionError if X and Y have different sample columns."""
        X, C, Y, _ = mock_genomic_data
        X_bad = X.copy()
        X_bad.columns = [f"wrong_sample_{i}" for i in range(X.shape[1])]

        with pytest.raises(AssertionError, match="same columns as Y"):
            structured_train_test_split(X_bad, C, Y)

    def test_assertion_overlapping_indices(self, mock_genomic_data):
        """Should raise AssertionError if X index (TFs) overlaps with Y index (genes)."""
        X, C, Y, _ = mock_genomic_data
        X_bad = X.copy()
        X_bad.index = Y.index[: len(X)]  # Force overlapping names

        with pytest.raises(AssertionError, match="must be disjoint"):
            structured_train_test_split(X_bad, C, Y)


# ---------------------------------------------------------------------------
# Tests for structured_cv_split
# ---------------------------------------------------------------------------
class TestStructuredCVSplit:

    def test_cv_split_basic_fold_structure(self, mock_genomic_data):
        """Verify fold count, index bounds, and train/val disjointness."""
        _, _, Y, _ = mock_genomic_data
        n_splits = 5

        folds = structured_cv_split(Y, n_splits=n_splits, random_state=42)

        assert len(folds) == n_splits

        for (train_row, train_col), (val_row, val_col) in folds:
            # Check row indices cover full dataset
            assert len(train_row) + len(val_row) == Y.shape[0]
            assert set(train_row).isdisjoint(set(val_row))

            # Check column indices cover full dataset
            assert len(train_col) + len(val_col) == Y.shape[1]
            assert set(train_col).isdisjoint(set(val_col))

    def test_cv_split_with_groups(self, mock_genomic_data):
        """Verify GroupKFold prevents row-group leakage across validation folds."""
        _, _, Y, row_groups = mock_genomic_data
        n_splits = 5

        folds = structured_cv_split(Y, n_splits=n_splits, row_groups=row_groups)

        for (train_row, _), (val_row, _) in folds:
            train_groups = set(row_groups[train_row])
            val_groups = set(row_groups[val_row])

            # Groups in train and validation must not overlap
            assert train_groups.isdisjoint(val_groups)
