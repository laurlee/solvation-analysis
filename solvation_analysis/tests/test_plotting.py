import numpy as np
import pytest
from solvation_analysis.plotting import *
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def test_plot_network_size_histogram(networking):
    fig = plot_network_size_histogram(networking)
    # fig.show()
    assert True


def test_plot_shell_size_histogram(run_solute):
    fig = plot_shell_size_histogram(run_solute)
    # fig.show()
    assert True


# compare_solvent_dicts tests
def test_compare_solvent_dicts_rename_exception(eax_solutes):
    # invalid solvents_to_plot because solvent names were already renamed to the generic "EAx" form
    # solvents_to_plot here references the former names of solvents, which is wrong
    # this test should handle an exception
    with pytest.raises(Exception):
        fig = compare_pairing(eax_solutes, rename_solvent_dict={"ea": "EAx", "fea": "EAx", "eaf": "EAx", "feaf": "EAx"},
                          solvents_to_plot=["pf6", "fec", "ea", "fea", "eaf", "feaf"], x_label="Species", y_label="Pairing", title="Graph")


def test_compare_solvent_dicts_sensitivity(eax_solutes):
    # solvent names are case-sensitive, so names in solvents_to_plot and rename_solvent_dict should be consistent
    # this test should handle an exception
    with pytest.raises(Exception):
        fig = compare_pairing(eax_solutes, rename_solvent_dict={"EA": "EAx", "fEA": "EAx", "EAf": "EAx", "fEAf": "EAx"},
                          solvents_to_plot=["PF6", "FEC", "EAx"], x_label="Species", y_label="Pairing", title="Graph")


# compare_pairing tests
def test_compare_pairing_default_eax(eax_solutes):
    # call compare_pairing with only one required argument
    # also tests how the code handles eax systems
    fig = compare_pairing(eax_solutes)
    assert len(fig.data) == 4
    # fig.show()


def test_compare_pairing_case1(eax_solutes):
    # solvents_to_plot on x axis, each bar is a solute
    fig = compare_pairing(eax_solutes, solvents_to_plot=["fec", "pf6"], x_label="Species", y_label="Pairing", title="Bar Graph of Solvent Pairing")
    assert len(fig.data) == 4
    for bar in fig.data:
        assert set(bar.x) == {"fec", "pf6"}
    # fig.show()


def test_compare_pairing_case2(eax_solutes):
    # solutes on x axis, each bar is an element of solvents_to_plot
    fig = compare_pairing(eax_solutes, solvents_to_plot=["pf6", "fec"], x_label="Solute", y_label="Pairing", title="Bar Graph of Solvent Pairing", x_axis="solute")
    assert len(fig.data) == 2
    for bar in fig.data:
        assert set(bar.x) == {"feaf", "eaf", "fea", "ea"}
    # fig.show()


def test_compare_pairing_case3(eax_solutes):
    # solvents_to_plot on x axis, each line is a solute
    fig = compare_pairing(eax_solutes, solvents_to_plot=["pf6", "fec"], x_label="Solute", y_label="Pairing", title="Line Graph of Solvent Pairing",series=True)
    assert len(fig.data) == 4
    for line in fig.data:
        assert set(line.x) == {"fec", "pf6"}
    # fig.show()


def test_compare_pairing_case4(eax_solutes):
    # solutes on x axis, each line is an element of solvents_to_plot
    fig = compare_pairing(eax_solutes, solvents_to_plot=["pf6", "fec"], x_label="Solute", y_label="Pairing", title="Line Graph of Solvent Pairing", x_axis="solute", series=True)
    assert len(fig.data) == 2
    for line in fig.data:
        assert set(line.x) == {"feaf", "eaf", "fea", "ea"}
    # fig.show()


def test_compare_pairing_switch_solvents_to_plot_order(eax_solutes):
    # same test as test_compare_pairing_case4, except order for solvents_to_plot is switched
    fig = compare_pairing(eax_solutes, solvents_to_plot=["fec", "pf6"], x_label="Solute", y_label="Pairing", title="Line Graph of Solvent Pairing", x_axis="solute", series=True)
    assert len(fig.data) == 2
    for line in fig.data:
        assert set(line.x) == {"feaf", "eaf", "fea", "ea"}
    # fig.show()


def test_compare_pairing_rename_solvent_dict(eax_solutes):
    # rename solvent names into the generic "EAx" form
    fig = compare_pairing(eax_solutes, rename_solvent_dict={"ea": "EAx", "fea": "EAx", "eaf": "EAx", "feaf": "EAx"},
                          solvents_to_plot=["pf6", "fec", "EAx"], x_label="Species", y_label="Pairing", title="Bar Graph of Solvent Pairing")
    assert len(fig.data) == 4
    for bar in fig.data:
        assert set(bar.x) == {"pf6", "fec", "EAx"}
    # fig.show()


# compare_coordination_numbers tests
def test_compare_coordination_numbers_default_eax(eax_solutes):
    # call compare_coordination_numbers with only one required argument
    # also tests how the code handles eax systems
    fig = compare_coordination_numbers(eax_solutes)
    assert len(fig.data) == 4
    # fig.show()


def test_compare_coordination_numbers_case1(eax_solutes):
    # solvents_to_plot on x axis, each bar is a solute
    fig = compare_coordination_numbers(eax_solutes, solvents_to_plot=["fec", "pf6"], x_label="Species", y_label="Coordination",
                          title="Bar Graph of Coordination Numbers")
    assert len(fig.data) == 4
    for bar in fig.data:
        assert set(bar.x) == {"fec", "pf6"}
    # fig.show()


def test_compare_coordination_numbers_case2(eax_solutes):
    # solutes on x axis, each bar is an element of solvents_to_plot
    fig = compare_coordination_numbers(eax_solutes, solvents_to_plot=["pf6", "fec"], x_label="solute", y_label="Coordination",
                          title="Bar Graph of Coordination Numbers", x_axis="solute")
    assert len(fig.data) == 2
    for bar in fig.data:
        assert set(bar.x) == {"feaf", "eaf", "fea", "ea"}
    # fig.show()


def test_compare_coordination_numbers_case3(eax_solutes):
    # solvents_to_plot on x axis, each line is a solute
    fig = compare_coordination_numbers(eax_solutes, solvents_to_plot=["pf6", "fec"], x_label="solute", y_label="Coordination",
                          title="Line Graph of Coordination Numbers", series=True)
    assert len(fig.data) == 4
    for line in fig.data:
        assert set(line.x) == {"fec", "pf6"}
    # fig.show()


def test_compare_coordination_numbers_case4(eax_solutes):
    # solutes on x axis, each line is an element of solvents_to_plot
    fig = compare_coordination_numbers(eax_solutes, solvents_to_plot=["pf6", "fec"], x_label="solute", y_label="Coordination",
                          title="Line Graph of Coordination Numbers", x_axis="solute", series=True)
    assert len(fig.data) == 2
    for line in fig.data:
        assert set(line.x) == {"feaf", "eaf", "fea", "ea"}
    # fig.show()


# compare_residence_times tests
def test_compare_residence_times_res_type_exception(eax_solutes):
    # this test should handle an exception relating to the acceptable arguments for res_type
    with pytest.raises(Exception):
        fig = compare_residence_times(eax_solutes, res_type="residence time", solvents_to_plot=["fec", "pf6"],
                                      x_label="Species",
                                      y_label="Residence Times",
                                      title="Bar Graph of Residence Times")


def test_compare_residence_times_instantiation_exception(eax_solutes):
    # this test should handle an exception relating to whether the Residence analysis class is instantiated
    with pytest.raises(Exception):
        fig = compare_residence_times(eax_solutes, res_type="residence_times", solvents_to_plot=["fec", "pf6"],
                                      x_label="Species",
                                      y_label="Residence Times",
                                      title="Bar Graph of Residence Times")


def test_plot_rdfs_case1_no_clasp(iba_small_solute):
    fig = plot_rdfs(iba_small_solute, "atom solute")
    fig.show()


def test_plot_rdfs_case2_no_clasp(iba_small_solute):
    fig = plot_rdfs(iba_small_solute, "solvent")
    fig.show()


def test_plot_rdfs_atom_solute_clasp_x_axis(iba_small_solute):
    fig = plot_rdfs(iba_small_solute, "atom solute", "x")
    fig.show()


def test_plot_rdfs_solvent_clasp_x_axis(iba_small_solute):
    fig = plot_rdfs(iba_small_solute, "solvent", "x")
    fig.show()


def test_plot_rdfs_atom_solute_clasp_y_axis(iba_small_solute):
    fig = plot_rdfs(iba_small_solute, "atom solute", "y")
    fig.show()

def test_plot_rdfs_solvent_clasp_y_axis(iba_small_solute):
    fig = plot_rdfs(iba_small_solute, "solvent", "y")
    fig.show()

