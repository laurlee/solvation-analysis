"""
========
Plotting
========
:Authors: Orion Cohen and Lauren Lee
:Year: 2023
:Copyright: GNU Public License v3

The plotting functions are a convenient way to visualize data by taking solutions
as their input and generating a Plotly.Figure object.
"""

import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import matplotlib
from copy import deepcopy

import numpy as np
import pandas as pd


# single solution
def plot_network_size_histogram(networking):
    """
    Returns a histogram of network sizes.

    Parameters
    ----------
    networking : Networking

    Returns
    -------
    fig : Plotly.Figure

    """
    network_sizes = networking.network_sizes
    sums = network_sizes.sum(axis=0)
    total_networks = sums.sum()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=sums.index, y=sums.values / total_networks))
    fig.update_layout(xaxis_title_text="Network Size", yaxis_title_text="Fraction of All Networks",
                      title="Histogram of Network Sizes")
    fig.update_xaxes(type="category")
    return fig


def plot_shell_size_histogram(solution):
    """
    Returns a histogram of shell sizes.

    Parameters
    ----------
    solution : Solution

    Returns
    -------
    fig : Plotly.Figure

    """
    speciation_data = solution.speciation.speciation_data
    speciation_data["total"] = speciation_data.sum(axis=1)
    sums = speciation_data.groupby("total").sum()
    fig = go.Figure()
    totals = sums.T.sum()
    for column in sums.columns:
        fig.add_trace(go.Bar(x=sums.index.values, y=sums[column].values / totals, name=column))
    fig.update_layout(xaxis_title_text="Shell Size", yaxis_title_text="Fraction of Total Molecules",
                      title="Fraction of Solvents in Shells of Different Sizes")
    fig.update_xaxes(type="category")
    return fig


# multiple solutions
def compare_solvent_dicts(property_dict, rename_solvent_dict, solvents_to_plot, legend_label, x_axis="species",
                          series=False):
    # generalist plotter, this can plot either bar or line charts of the same data
    """

    Parameters
    ----------
    property_dict : dict of {str: dict}
        a dictionary with the solution name as keys and a dict of {str: float} as values, where each string
        is the name of the species of each solution and each float is the value of the property of interest
    rename_solvent_dict : dict of {str: str}, where the keys are strings of solvent names and the values are
        strings of a more generic name for the solvent (i.e. {"EAf" : "EAx", "fEAf" : "EAx"})
    solvents_to_plot : list of strings of solvent names that are common to all systems in question,
        graphed in the order that is passed into the function. Names of solutions are first swapped with the
        more generic name, if rename_solvent_dict is specified, before filtering for solution names
        specified by solvents_to_plot. In order for solvents_to_plot to execute properly, any solution names
        affected by the swap with rename_solvent_dict must be referenced by the generic name in solvents_to_plot.
    legend_label : title of legend as a string
    x_axis : a string specifying "species" or "solution" to be graphed on the x_axis
    series : Boolean (False for a bar graph; True for a line graph)

    Returns
    -------
    fig : Plotly.Figure (generic plot)

    """
    property_dict = deepcopy(property_dict)
    # coerce solutions to a common name
    for solution_name in rename_solvent_dict:
        if solution_name in property_dict:
            common_name = rename_solvent_dict[solution_name]
            # remove the solution name from the properties dict and rename to the common name
            solution_property_value = property_dict[solution_name].pop(solution_name)
            property_dict[solution_name][common_name] = solution_property_value

    # filter out components of solution to only include those in solvents_to_plot
    if solvents_to_plot:
        for solution_name, solution_dict in property_dict.items():
            new_property_dict = {}
            for solvent_name in solvents_to_plot:
                solvent_dict = property_dict[solution_name].get(solvent_name)
                if solvent_dict is None:
                    raise Exception("Invalid value of solvents_to_plot. \n solvents_to_plot: " +
                                    str(solvents_to_plot) + "\n Valid options for solvents_to_plot: " +
                                    str(list(property_dict[solution_name].keys()))) from None
                new_property_dict[solvent_name] = solvent_dict
            property_dict[solution_name] = new_property_dict

    # generate figure and make a DataFrame of the data
    fig = go.Figure()
    df = pd.DataFrame(data=property_dict.values())
    df.index = list(property_dict.keys())

    if series and x_axis == "species":
        # each solution is a line
        df = df.transpose()
        fig = px.line(df, x=df.index, y=df.columns, labels={"variable": legend_label})
        fig.update_xaxes(type="category")
    elif series and x_axis == "solute":
        # each species is a line
        fig = px.line(df, x=df.index, y=df.columns, labels={"variable": legend_label})
        fig.update_xaxes(type="category")
    elif not series and x_axis == "species":
        # each solution is a bar
        df = df.transpose()
        fig = px.bar(df, x=df.index, y=df.columns, barmode="group", labels={"variable": legend_label})
    elif not series and x_axis == "solute":
        # each species is a bar
        fig = px.bar(df, x=df.index, y=df.columns, barmode="group", labels={"variable": legend_label})
    return fig


def compare_free_solvents(solutions):
    # this should be a grouped vertical bar chart or a line chart
    # 1.0 should be marked and annotated with a dotted line
    fig = compare_solvent_dicts()
    return


def compare_pairing(solutions, rename_solvent_dict=None, solvents_to_plot=None, x_label="Solvent", y_label="Pairing",
                    title="Graph of Pairing Data", legend_label="Legend", **kwargs):
    # this should be a grouped vertical bar chart or a line chart
    # 1.0 should be marked and annotated with a dotted line
    """
    Compares the pairing of multiple solutions.
    Parameters
    ----------
    solutions : dict of {str: Solution}, where the key is the name of the Solution object and the values are
        the Solution object
    rename_solvent_dict : dict of {str: str}, where the keys are strings of solvent names and the values are
        strings of a more generic name for the solvent (i.e. {"EAf" : "EAx", "fEAf" : "EAx"})
    solvents_to_plot : list of strings of solvent names that are common to all systems in question,
        graphed in the order that is passed into the function. Names of solutions are first swapped with the
        more generic name, if rename_solvent_dict is specified, before filtering for solution names
        specified by solvents_to_plot. In order for solvents_to_plot to execute properly, any solution names
        affected by the swap with rename_solvent_dict must be referenced by the generic name in solvents_to_plot.
    x_label : name of the x-axis as a string
    y_label : name of the y-axis as a string
    title : title of figure as a string
    legend_label : title of legend as a string
    kwargs : consists of the x_axis and series parameters
        x_axis : a string specifying "species" or "solution" to be graphed on the x_axis
        series : Boolean (False for a bar graph; True for a line graph)

    Returns
    -------
    fig : Plotly.Figure

    """
    rename_solvent_dict = rename_solvent_dict or {}
    pairing = {solution_name: solutions[solution_name].pairing.pairing_dict for solution_name in solutions}
    fig = compare_solvent_dicts(pairing, rename_solvent_dict, solvents_to_plot, legend_label, **kwargs)
    fig.update_layout(xaxis_title_text=x_label.title(), yaxis_title_text=y_label.title(), title=title.title())
    return fig


def compare_coordination_numbers(solutions, rename_solvent_dict=None, solvents_to_plot=None, x_label="Solvent",
                                 y_label="Coordination", title="Graph of Coordination Data", legend_label="Legend",
                                 **kwargs):
    """
    Compares the coordination numbers of multiple solutions.

    Parameters
    ----------
    solutions : dict of {str: Solution}, where the key is the name of the Solution object and the values are
        the Solution object
    rename_solvent_dict : dict of {str: str}, where the keys are strings of solvent names and the values are
        strings of a more generic name for the solvent (i.e. {"EAf" : "EAx", "fEAf" : "EAx"})
    solvents_to_plot : list of strings of solvent names that are common to all systems in question,
        graphed in the order that is passed into the function. Names of solutions are first swapped with the
        more generic name, if rename_solvent_dict is specified, before filtering for solution names
        specified by solvents_to_plot. In order for solvents_to_plot to execute properly, any solution names
        affected by the swap with rename_solvent_dict must be referenced by the generic name in solvents_to_plot.
    x_label : name of the x-axis as a string
    y_label : name of the y-axis as a string
    title : title of figure as a string
    legend_label : title of legend as a string
    kwargs : consists of the x_axis and series parameters
        x_axis : a string specifying "species" or "solution" to be graphed on the x_axis
        series : Boolean (False for a bar graph; True for a line graph)

    Returns
    -------
    fig : Plotly.Figure

    """
    rename_solvent_dict = rename_solvent_dict or {}
    coordination = {solution_name: solutions[solution_name].coordination.cn_dict for solution_name in solutions}
    fig = compare_solvent_dicts(coordination, rename_solvent_dict, solvents_to_plot, legend_label, **kwargs)
    fig.update_layout(xaxis_title_text=x_label.title(), yaxis_title_text=y_label.title(), title=title.title())
    return fig


def compare_residence_times(solutions, res_type="residence_times_fit", rename_solvent_dict=None, solvents_to_plot=None,
                            x_label="Solvent", y_label="Residence Time", title="Graph of Residence Time Data",
                            legend_label="Legend", **kwargs):
    """
    Compares the residence times of multiple solutions.

    Parameters
    ----------
    solutions : dict of {str: Solution}, where the key is the name of the Solution object and the values are
        the Solution object
    res_type : a string that is either "residence_times" or residence_times_fit"
    rename_solvent_dict : dict of {str: str}, where the keys are strings of solvent names and the values are
        strings of a more generic name for the solvent (i.e. {"EAf" : "EAx", "fEAf" : "EAx"})
    solvents_to_plot : list of strings of solvent names that are common to all systems in question,
        graphed in the order that is passed into the function. Names of solutions are first swapped with the
        more generic name, if rename_solvent_dict is specified, before filtering for solution names
        specified by solvents_to_plot. In order for solvents_to_plot to execute properly, any solution names
        affected by the swap with rename_solvent_dict must be referenced by the generic name in solvents_to_plot.
    x_label : name of the x-axis as a string
    y_label : name of the y-axis as a string
    title : title of figure as a string
    legend_label : title of legend as a string
    kwargs : consists of the x_axis and series parameters
        x_axis : a string specifying "species" or "solution" to be graphed on the x_axis
        series : Boolean (False for a bar graph; True for a line graph)

    Returns
    -------
    fig : Plotly.Figure

    """
    #
    if res_type not in ["residence_times", "residence_times_fit"]:
        raise ValueError("res_type must be either \"residence_times\" or \"residence_times_fit\"")

    res_time = {}
    for solution_name, solution in solutions.items():
        if not hasattr(solution, "residence"):
            raise ValueError("Solution's Residence analysis class is not instantiated.")
        res_time[solution_name] = getattr(solution.residence, res_type)

    rename_solvent_dict = rename_solvent_dict or {}
    fig = compare_solvent_dicts(res_time, rename_solvent_dict, solvents_to_plot, legend_label, **kwargs)
    fig.update_layout(xaxis_title_text=x_label.title(), yaxis_title_text=y_label.title(), title=title.title())
    return fig


def compare_rdfs(solutions, atoms):
    # can atom groups be matched to solutions / universes behind the scenes?
    # yes we can use atom.u is universe
    return


def plot_rdfs(solute, x_axis, clasp=None):
    rdf_data = solute.rdf_data
    df = pd.DataFrame()
    dataframes = {}
    for atom_solute_name in rdf_data:
        for solvent in rdf_data[atom_solute_name]:
            temp = pd.DataFrame(data=rdf_data[atom_solute_name][solvent]).transpose()
            temp.columns = ["bins", "rdf"]
            temp["solvent"] = solvent
            temp["atom solute"] = atom_solute_name
            dataframes[(atom_solute_name, solvent)] = temp[["bins", "rdf"]]
            df = pd.concat([df, temp])

    atom_solutes, solvents = zip(*dataframes.keys());

    atom_solutes = set(atom_solutes)
    solvents = set(solvents)

    # TODO: check subset of df
    fig = go.Figure()
    # 2x3 grid --> 2x1 grid, each graph has three traces
    if clasp == "x" and x_axis == "atom solute":
        fig = make_subplots(rows=len(atom_solutes), cols=1, x_title="Atom Solute")
        r = 1
        for atom_solute in atom_solutes:
            temp = df.loc[df["atom solute"] == atom_solute]
            fig.add_trace(go.Scatter(x=temp["bins"], y=temp["rdf"]), row=r, col=1)
            fig.update_xaxes(title_text=atom_solute, row=r, col=1)
            r += 1
    # 3x2 grid --> 3x1 grid, each graph has two traces
    elif clasp == "x" and x_axis == "solvent":
        fig = make_subplots(rows=1, cols=len(atom_solutes), x_title="Solvent")
        c = 1
        for solvent in solvents:
            temp = df.loc[df["solvent"] == solvent]
            fig.add_trace(go.Scatter(x=temp["bins"], y=temp["rdf"]), row=1, col=c)
            fig.update_xaxes(title_text=solvent, row=1, col=c)
            c += 1
    # 2x3 grid --> 1x3 grid, each graph has two traces
    elif clasp == "y" and x_axis == "atom solute":
        fig = make_subplots(rows=len(solvents), cols=len(atom_solutes), x_title="Atom Solute")
        c = 1
        for atom_solute in atom_solutes:
            temp = df.loc[df["atom solute"] == atom_solute]
            fig.add_trace(go.Scatter(x=temp["bins"], y=temp["rdf"]), row=1, col=c)
            fig.update_xaxes(title_text=atom_solute, row=1, col=c)
            c += 1
    # 3x2 grid --> 1x2 grid, each graph has three traces
    elif clasp == "y" and x_axis == "solvent":
        print("need to work on this")
    elif x_axis == "atom solute":
        fig = make_subplots(rows=len(solvents), cols=len(atom_solutes), shared_xaxes=True, shared_yaxes=True, x_title="Atom Solute", y_title="Solvent")

        r = 1
        for solvent in solvents:
            c = 1
            for atom_solute in atom_solutes:
                data = dataframes[(atom_solute, solvent)]
                fig.add_trace(go.Scatter(x=data["bins"], y=data["rdf"]), row=r, col=c)
                fig.update_xaxes(title_text=atom_solute, row=r, col=c)
                fig.update_yaxes(title_text=solvent, row=r, col=c)
                c += 1
            r += 1
    elif x_axis == "solvent":
        fig = make_subplots(rows=len(atom_solutes), cols=len(solvents), shared_xaxes=True, shared_yaxes=True, x_title="Solvent", y_title="Atom Solute")

        r = 1
        for atom_solute in atom_solutes:
            c = 1
            for solvent in solvents:
                data = dataframes[(atom_solute, solvent)]
                fig.add_trace(go.Scatter(x=data["bins"], y=data["rdf"]), row=r, col=c)
                fig.update_xaxes(title_text=solvent, row=r, col=c)
                fig.update_yaxes(title_text=atom_solute, row=r, col=c)
                c += 1
            r += 1

    fig.update_layout(showlegend=False, template="simple_white")
    return fig
