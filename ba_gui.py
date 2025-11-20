from __future__ import annotations

# pylint: disable=no-name-in-module
from pathlib import Path
from typing import List

import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QFileDialog,
    QMessageBox,
    QListWidget,
    QComboBox,
    QCheckBox,
    QTextEdit,
    QLabel,
    QPushButton,
    QLineEdit,
    QGridLayout,
    QHBoxLayout,
    QGroupBox,
    QAbstractItemView,
)

from ba_core import run_ba_pipeline
from ba_plot import plot_ba_results


class BAGui(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BACalc - Advanced")
        self.resize(1050, 750)
        self.setMinimumSize(950, 650)

        self.files: List[Path] = []
        self.df: pd.DataFrame | None = None

        self.out_dir = Path.cwd() / "output_results"

        # widgets
        self.files_list = QListWidget()
        self.btn_add_csv = QPushButton("Add CSV…")
        self.btn_clear = QPushButton("Clear List")
        self.btn_load_merge = QPushButton("Load / Merge")

        self.age_box = QComboBox()
        self.split_box = QComboBox()
        self.chk_map = QCheckBox(
            "Compute MAP from Systolic_BP and Diastolic_BP (creates 'MAP')"
        )

        self.biom_list = QListWidget()

        self.biom_list.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )

        self.out_edit = QLineEdit(str(self.out_dir))
        self.btn_browse_out = QPushButton("Browse…")
        self.chk_save_plots = QCheckBox("Save plots after run")
        self.chk_save_plots.setChecked(True)

        self.btn_run = QPushButton("Run BA Estimation")
        self.btn_plot = QPushButton("Re-Plot Results")

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)

        self.btn_about = QPushButton("About…")
        self.btn_quit = QPushButton("Quit")

        self.wire_signals()
        self.layout()

    def wire_signals(self):
        self.btn_add_csv.clicked.connect(self.add_csv)
        self.btn_clear.clicked.connect(self.clear_files)
        self.btn_load_merge.clicked.connect(self.load_merged_df)

        self.btn_browse_out.clicked.connect(self.browse_out_dir)
        self.btn_run.clicked.connect(self.run_ba)
        self.btn_plot.clicked.connect(self.plot_results)
        self.btn_about.clicked.connect(self.show_about)
        self.btn_quit.clicked.connect(self.close)

    def layout(self):
        root = QWidget()
        self.setCentralWidget(root)
        gl = QGridLayout(root)
        gl.setContentsMargins(10, 10, 10, 10)
        gl.setHorizontalSpacing(10)
        gl.setVerticalSpacing(8)

        # file loader
        grp_top = QGroupBox("1) Load CSV file(s)")
        gl.addWidget(grp_top, 0, 0, 1, 2)
        top = QGridLayout(grp_top)
        top.addWidget(self.btn_add_csv, 0, 0)
        top.addWidget(self.btn_clear, 0, 1)
        top.addWidget(self.files_list, 0, 2)
        top.addWidget(self.btn_load_merge, 0, 3)
        top.setColumnStretch(2, 1)

        grp_mid = QGroupBox("2) Select columns")
        gl.addWidget(grp_mid, 1, 0, 1, 2)
        mid = QGridLayout(grp_mid)

        mid.addWidget(QLabel("Chronological Age column:"), 0, 0)
        mid.addWidget(self.age_box, 0, 1)

        mid.addWidget(QLabel("Optional binary split column:"), 1, 0)
        mid.addWidget(self.split_box, 1, 1)

        mid.addWidget(self.chk_map, 2, 0, 1, 2)

        mid.addWidget(QLabel("Biomarker columns (multi-select):"), 3, 0, 1, 2)
        mid.addWidget(self.biom_list, 4, 0, 1, 2)
        mid.setRowStretch(4, 1)

        # output and run in the gui
        grp_bot = QGroupBox("3) Output, Plots & Actions")
        gl.addWidget(grp_bot, 2, 0, 1, 2)
        bot = QGridLayout(grp_bot)

        bot.addWidget(QLabel("Output directory:"), 0, 0)
        bot.addWidget(self.out_edit, 0, 1)
        bot.addWidget(self.btn_browse_out, 0, 2)
        bot.addWidget(self.chk_save_plots, 1, 0, 1, 2)
        bot.addWidget(self.btn_run, 1, 2)
        bot.addWidget(self.btn_plot, 1, 3)
        bot.setColumnStretch(1, 1)

        # logbox
        gl.addWidget(self.log_box, 3, 0, 1, 2)

        # footer section
        footer = QHBoxLayout()
        footer.addWidget(self.btn_about)
        footer.addStretch(1)
        footer.addWidget(self.btn_quit)
        gl.addLayout(footer, 4, 0, 1, 2)

    # csv file reader
    def add_csv(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select CSV file(s)", "", "CSV files (*.csv);;All files (*)"
        )
        if not paths:
            return
        for p in paths:
            pth = Path(p)
            self.files.append(pth)
            self.files_list.addItem(str(pth))

    def clear_files(self):
        self.files.clear()
        self.files_list.clear()
        self.df = None
        self.refresh_columns([])

    def load_merged_df(self):
        if not self.files:
            QMessageBox.warning(self, "No files", "Please add at least one CSV.")
            return

        dfs = []
        errors = []
        for p in self.files:
            try:
                dfs.append(pd.read_csv(p))
            except Exception as e:
                errors.append(f"{p.name}: {e}")

        if not dfs:
            QMessageBox.critical(
                self, "Load error", "No CSVs could be loaded.\n" + "\n".join(errors)
            )
            return

        self.df = pd.concat(dfs, ignore_index=True, sort=False)
        self.log(f"Loaded rows: {len(self.df)} from {len(dfs)} file(s).")
        if errors:
            self.log("Some files failed to load:\n" + "\n".join(errors))

        cols = list(self.df.columns)
        self.refresh_columns(cols)
        self.autoset_age_column(cols)
        self.populate_binary_candidates()
        self.update_map_checkbox()

    def refresh_columns(self, cols):
        self.age_box.clear()
        self.age_box.addItems(cols)
        self.split_box.clear()
        self.split_box.addItems(cols)
        self.biom_list.clear()
        for c in cols:
            self.biom_list.addItem(c)

    def autoset_age_column(self, cols):
        for cand in ["Age", "age", "AGE", "Chronological_Age", "CA"]:
            if cand in cols:
                idx = self.age_box.findText(cand, Qt.MatchExactly)
                if idx >= 0:
                    self.age_box.setCurrentIndex(idx)
                return

    def populate_binary_candidates(self):
        if self.df is None:
            return
        candidates = []
        for c in self.df.columns:
            s = self.df[c].dropna()
            uniq = s.unique()
            if len(uniq) == 2:
                candidates.append(c)
        # binary vars show
        self.split_box.clear()
        self.split_box.addItem("")  # empty fir no split
        self.split_box.addItems([str(x) for x in candidates])

    def update_map_checkbox(self):
        if self.df is None:
            self.chk_map.setChecked(False)
            self.chk_map.setEnabled(False)
            return
        has_sbp = "Systolic_BP" in self.df.columns
        has_dbp = "Diastolic_BP" in self.df.columns
        self.chk_map.setEnabled(has_sbp and has_dbp)
        if not (has_sbp and has_dbp):
            self.chk_map.setChecked(False)

    # filediag
    def browse_out_dir(self):
        d = QFileDialog.getExistingDirectory(
            self, "Choose output directory", str(self.out_dir)
        )
        if d:
            self.out_dir = Path(d)
            self.out_edit.setText(str(self.out_dir))

    def run_ba(self):
        try:
            if self.df is None or self.df.empty:
                QMessageBox.warning(self, "No data", "Please load CSV(s) first.")
                return

            age = self.age_box.currentText().strip()
            if not age:
                QMessageBox.warning(
                    self, "Missing age", "Please choose the chronological age column."
                )
                return

            selected_items = self.biom_list.selectedItems()
            if not selected_items:
                QMessageBox.warning(
                    self,
                    "No biomarkers",
                    "Please select one or more biomarker columns.",
                )
                return
            biom_cols = [it.text() for it in selected_items]

            df_run = self.df.copy()
            if self.chk_map.isChecked():
                if ("Systolic_BP" not in df_run.columns) or (
                    "Diastolic_BP" not in df_run.columns
                ):
                    QMessageBox.critical(
                        self,
                        "MAP error",
                        "Systolic_BP and Diastolic_BP are required to compute MAP.",
                    )
                    return
                df_run["MAP"] = (
                    df_run["Systolic_BP"] / 3.0 + 2.0 * df_run["Diastolic_BP"] / 3.0
                )
                if "MAP" not in biom_cols:
                    biom_cols = ["MAP"] + biom_cols

            missing = [c for c in biom_cols if c not in df_run.columns]
            if missing:
                QMessageBox.critical(
                    self,
                    "Missing columns",
                    "These biomarkers are missing:\n" + "\n".join(missing),
                )
                return

            split = self.split_box.currentText().strip() or None

            self.log("Starting pipeline...")
            outputs = run_ba_pipeline(
                df=df_run,
                age_col=age,
                biom_cols=biom_cols,
                split_col=split,
                out_dir=Path(self.out_edit.text()).expanduser(),
                log=self.log,
            )

            self.log("\nSaved files:")
            for k, p in outputs.items():
                self.log(f"  {k}: {p}")

            if self.chk_save_plots.isChecked():
                self.log("\nGenerating advanced plots...")
                plot_info = plot_ba_results(
                    outputs["predictions"],
                    Path(self.out_edit.text()).expanduser(),
                    log=self.log,
                )
                self.log(f"Plots saved under: {plot_info['plots_dir']}")

            QMessageBox.information(self, "Done", "BA estimation complete.")

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            self.log(f"ERROR: {e}")
            # print traceback for debugging
            import traceback

            traceback.print_exc()

    def plot_results(self):
        try:
            pred_csv = Path(self.out_edit.text()).expanduser() / "ba_predictions.csv"
            if not pred_csv.exists():
                QMessageBox.warning(
                    self, "Not found", f"Could not find {pred_csv}. Run BA first."
                )
                return
            self.log("\nGenerating plots from existing predictions...")
            plot_info = plot_ba_results(
                pred_csv, Path(self.out_edit.text()).expanduser(), log=self.log
            )
            self.log(f"Plots saved under: {plot_info['plots_dir']}")
            QMessageBox.information(
                self, "Plots saved", f"Plots saved to:\n{plot_info['plots_dir']}"
            )
        except Exception as e:
            QMessageBox.critical(self, "Plot error", str(e))
            self.log(f"ERROR (plot): {e}")

    def show_about(self):
        QMessageBox.information(
            self,
            "About",
            "Dynamic and Dataset Agnostic Biological Age Calculator (BACalc)\n"
            "PCA > BAS > T-scale > Dubina-corrected BAc\n\n"
            "Developed by Mehrdad S. Beni, Haipeng Liu and Gary Tse -- 2025",
        )

    def log(self, msg: str):
        self.log_box.append(str(msg))
        self.log_box.ensureCursorVisible()


def main():
    import sys

    app = QApplication(sys.argv)
    gui = BAGui()
    gui.show()
    sys.exit(app.exec())
