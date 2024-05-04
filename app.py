import marimo

__generated_with = "0.4.10"
app = marimo.App(layout_file="layouts/app.grid.json")


@app.cell
def __(execute_button, mo, selection_switch, sys_select):
    if not selection_switch.value:
        _panel = mo.hstack([sys_select, execute_button])
    else:
        _panel = mo.right(execute_button)
    _panel
    return


@app.cell
def __(selection_switch):
    selection_switch
    return


@app.cell
def __(selection_switch):
    selection_switch.value
    return


@app.cell
def __(table_view):
    table_view.value
    return


@app.cell
def __(selected_siteid):
    selected_siteid
    return


@app.cell
def __(selected_sysid):
    selected_sysid
    return


@app.cell
def __(cache_list, selection_switch, sys_select, system_list):
    if not selection_switch.value:
        if len(system_list.value) != 0:
            selected_siteid = system_list.value["Site ID"].iloc[0]
            selected_sysid = sys_select.value
        else:
            selected_siteid = "TADBC1078041"
            selected_sysid = 0
    else:
        if len(cache_list.value) != 0:
            selected_siteid = cache_list.value["Site ID"].iloc[0]
            selected_sysid = cache_list.value["System Index"].iloc[0]
        else:
            selected_siteid = "TADBC1078041"
            selected_sysid = 0
    return selected_siteid, selected_sysid


@app.cell
def __(table_view):
    table_view
    return


@app.cell
def __(dh, mo):
    with mo.redirect_stdout():
        dh.report()
    return


@app.cell
def __(capacity, clipping, daily, heatmaps, losses, mo, timeshift):
    mo.ui.tabs(
        {
            "data viewer": heatmaps,
            "losses": losses,
            "daily data quality": daily,
            "time shifts": timeshift,
            "capacity changes": capacity,
            "clipping analysis": clipping,
        }
    )
    return


@app.cell
def __():
    import marimo as mo
    import matplotlib.pyplot as plt

    # plt.rcParams["figure.dpi"] = 200
    # plt.rcParams["savefig.dpi"] = 200
    import numpy as np
    import pandas as pd
    import os
    import sys
    from pathlib import Path
    from contextlib import contextmanager
    from io import StringIO
    from functools import cache
    from solardatatools import DataHandler
    from solardatatools.dataio import load_redshift_data


    @contextmanager
    def capture_stdout():
        # Save the original stdout
        original_stdout = sys.stdout

        # Create a StringIO object to capture the output
        captured_output = StringIO()
        sys.stdout = captured_output

        try:
            # Yield control to the code within the 'with' block
            yield captured_output
        finally:
            # Restore the original stdout
            sys.stdout = original_stdout
    return (
        DataHandler,
        Path,
        StringIO,
        cache,
        capture_stdout,
        contextmanager,
        load_redshift_data,
        mo,
        np,
        os,
        pd,
        plt,
        sys,
    )


@app.cell
def __(keys, mo):
    site_select = mo.ui.dropdown(
        keys["Site ID"], label="site key", value="TADBC1078041"
    )
    return site_select,


@app.cell
def __(keys, mo, np, selection_switch, system_list):
    if not selection_switch.value:
        if len(system_list.value) != 0:
            _siteid = system_list.value["Site ID"].iloc[0]
        else:
            _siteid = "TADBC1078041"
        num_sys = int(keys[keys["Site ID"] == _siteid]["System Count"].iloc[0])
        _a = {f"{val:02}": val for val in np.arange(num_sys)}
        sys_select = mo.ui.dropdown(_a, label="system key", value="00")
    return num_sys, sys_select


@app.cell
def __(DataHandler, cache, capture_stdout, load_redshift_data, mo, os, pd):
    get_cache_history, set_cache_history = mo.state(
        pd.DataFrame(columns=["Site ID", "System Index"])
    )


    @cache
    def process_file(site_key, system_key):
        mo.output.replace(
            mo.md(
                f"""
        ## Loading data...
        site: {site_key}, system: {system_key}
        """
            )
        )
        query = {
            "siteid": site_key,
            "api_key": os.environ.get("REDSHIFT_API_KEY"),
            "sensor": int(system_key),
        }
        df = load_redshift_data(**query)
        mo.output.replace(
            mo.md(
                f"""
        ## Running SDT pipeline...
        site: {site_key}, system: {system_key}\n
        check console for progress
        """
            )
        )
        dh = DataHandler(df, convert_to_ts=True)
        # dh.fix_dst()
        with capture_stdout() as captured:
            try:
                dh.run_pipeline(fix_shifts=True)
            except:
                pass
        mo.output.replace(
            mo.md(
                f"""
        ## Running loss factor analysis...
        site: {site_key}, system: {system_key}\n
        ncheck console for progress
        """
            )
        )
        if dh.num_days > 365 * 1.5:
            dh.run_loss_factor_analysis()
        mo.output.replace(mo.md("## Done, and results cached!"))
        return dh, captured
    return get_cache_history, process_file, set_cache_history


@app.cell
def __():
    dh_container = [None]
    return dh_container,


@app.cell
def __():
    ids_container = [None, None]
    return ids_container,


@app.cell
def __(ids_container):
    ids_container
    return


@app.cell
def __(
    dh_container,
    get_cache_history,
    get_execute_state,
    ids_container,
    mo,
    process_file,
    selected_siteid,
    selected_sysid,
    set_cache_history,
    set_execute_state,
):
    if get_execute_state():
        _rdf = get_cache_history()
        is_in_cache = (
            (_rdf["Site ID"] == selected_siteid)
            & (_rdf["System Index"] == selected_sysid)
        ).any()
        if not is_in_cache:
            _rdf.loc[len(_rdf)] = [selected_siteid, selected_sysid]
            set_cache_history(_rdf)
        dh, captured = process_file(
            selected_siteid,
            selected_sysid,
        )
        dh_container[0] = dh
        ids_container[0] = selected_siteid
        ids_container[1] = selected_sysid
        executed_siteid = selected_siteid
        executed_sysid = selected_sysid
        set_execute_state(False)
    else:
        dh = dh_container[0]
        executed_siteid = ids_container[0]
        executed_sysid = ids_container[1]
    if not dh:
        pass
    else:
        mo.output.replace(
            mo.md(
                f"""
        ## Done, and results cached!
        site: {executed_siteid}, system: {executed_sysid}
        """
            )
        )
    return captured, dh, executed_siteid, executed_sysid, is_in_cache


@app.cell
def __(dh):
    dse = dh.loss_analysis
    return dse,


@app.cell
def __(mo):
    get_day, set_day = mo.state(0)
    return get_day, set_day


@app.cell
def __(dh, get_day, mo, set_day):
    start_day_select = mo.ui.slider(
        0,
        len(dh.day_index) - 1,
        1,
        label="start day",
        value=get_day(),
        on_change=set_day,
    )
    return start_day_select,


@app.cell
def __(mo):
    num_day_select = mo.ui.slider(1, 14, 1, value=5, label="number of days")
    return num_day_select,


@app.cell
def __(dh, get_day, mo, set_day):
    start_day_select2 = mo.ui.number(
        0,
        len(dh.day_index) - 1,
        1,
        label="start day",
        value=get_day(),
        on_change=set_day,
    )
    return start_day_select2,


@app.cell
def __(mo):
    get_execute_state, set_execute_state = mo.state(False)
    return get_execute_state, set_execute_state


@app.cell
def __(mo, set_execute_state):
    execute_button = mo.ui.button(
        label="execute",
        value=False,
        on_click=lambda _: set_execute_state(True),
    )
    return execute_button,


@app.cell
def __(Path, __file__, mo, pd):
    keys = pd.read_csv(
        Path(__file__).parent / "inputs" / "system_counts_per_site.csv"
    )
    keys.columns = ["Site ID", "System Count"]
    system_list = mo.ui.table(keys, page_size=8, selection="single")
    return keys, system_list


@app.cell
def __(get_cache_history, mo):
    cache_list = mo.ui.table(get_cache_history(), page_size=8, selection="single")
    return cache_list,


@app.cell
def __(cache_list, mo, system_list):
    table_view = mo.ui.tabs(
        {"System List": system_list, "Cache List": cache_list}, lazy=True
    )
    return table_view,


@app.cell
def __(data_switch, dh, get_day, plt):
    if not data_switch.value:
        with plt.rc_context({"figure.dpi": 200}):
            hmfig = dh.plot_heatmap("raw", figsize=(12, 5))
            plt.axvline(get_day(), color="yellow", ls="--", linewidth=1)
    else:
        with plt.rc_context({"figure.dpi": 200}):
            hmfig = dh.plot_heatmap("filled", figsize=(12, 5))
            plt.axvline(get_day(), color="yellow", ls="--", linewidth=1)
    return hmfig,


@app.cell
def __(
    data_switch,
    dh,
    get_day,
    hmfig,
    mo,
    num_day_select,
    start_day_select,
    start_day_select2,
):
    # with plt.rc_context({"figure.dpi": 200}):
    #     hmfig = dh.plot_heatmap("raw", figsize=(12, 5))
    # plt.axvline(get_day(), color="yellow", ls="--", linewidth=1)
    heatmaps = mo.vstack(
        [
            hmfig,
            mo.hstack(
                [data_switch, start_day_select, start_day_select2, num_day_select]
            ),
            dh.plot_daily_signals(
                start_day=get_day(),
                num_days=num_day_select.value,
                figsize=(12, 4),
            ),
        ]
    )
    return heatmaps,


@app.cell
def __(dse, mo, np, static_plots):
    try:
        _pie = static_plots["pie"]
        _waterfall = static_plots["waterfall"]
        _fig_decomp = static_plots["decomp"]
        _val = np.round(dse.degradation_rate, 2)
        losses = mo.vstack(
            [
                mo.center(mo.md(f"##estimated degradation rate: {_val:.02}%/yr")),
                mo.hstack([_pie, _waterfall]),
                mo.center(_fig_decomp),
            ]
        )
    except:
        losses = mo.md("## Loss analysis not run")
    return losses,


@app.cell
def __(dh, mo, static_plots):
    _report = dh.report(verbose=False, return_values=True)
    _qval = 100 * (1 - _report["quality score"])
    daily = mo.vstack(
        [
            mo.center(
                mo.md(f"##{_qval:.0f}% of days experienced a system outage")
            ),
            mo.center(static_plots["energy"]),
            mo.center(static_plots["density"]),
        ]
    )
    return daily,


@app.cell
def __(mo, static_plots):
    timeshift = mo.center(static_plots["timeshift"])
    return timeshift,


@app.cell
def __(dh, mo, static_plots):
    _num_clusters = len(set(dh.capacity_analysis.labels))
    capacity = mo.vstack(
        [
            mo.center(mo.md(f"##{_num_clusters} capacity levels detected")),
            mo.center(static_plots["capacity"]),
        ]
    )
    return capacity,


@app.cell
def __(dh, mo, static_plots):
    _report = dh.report(verbose=False, return_values=True)
    _cval = 100 * _report["clipped fraction"]
    clipping = mo.vstack(
        [
            mo.center(mo.md(f"##{_cval:.0f}% of days experienced clipping")),
            mo.center(static_plots["clipping"]),
        ]
    )
    return clipping,


@app.cell
def __(cache, dh, dse, np, plt):
    @cache
    def make_static_plots(site_key, system_key):
        try:
            _pie = dse.plot_pie()
            plt.figure()
            _waterfall = dse.plot_waterfall()
            plt.figure()
            _fig_decomp = dse.problem.plot_decomposition(
                exponentiate=True, figsize=(16, 8.5)
            )
            _ax = _fig_decomp.axes
            _ax[0].plot(
                np.arange(len(dse.energy_data))[~dse.use_ixs],
                dse.energy_model[-1, ~dse.use_ixs],
                color="red",
                marker=".",
                ls="none",
            )
            _ax[0].set_title("weather and system outages")
            _ax[1].set_title("capacity changes")
            _ax[2].set_title("soiling")
            _ax[3].set_title("degradation")
            _ax[4].set_title("baseline")
            _ax[5].set_title("measured energy (green) and model minus weather")
            plt.tight_layout()
        except:
            _pie = None
            _waterfall = None
            _fig_decomp = None
        daily_energy = dh.plot_daily_energy(flag="bad", figsize=(12, 3))
        daily_density = dh.plot_density_signal(
            flag="bad", show_fit=True, figsize=(12, 3)
        )
        plt.figure()
        tsh = dh.plot_time_shift_analysis_results()
        implemented = (
            dh.time_shift_analysis.correction_estimate / -60
            + dh.time_shift_analysis.baseline
        )
        plt.plot(dh.day_index, implemented, color="blue", label="implemented")
        plt.legend()
        out = {
            "pie": _pie,
            "waterfall": _waterfall,
            "decomp": _fig_decomp,
            "energy": daily_energy,
            "density": daily_density,
            "capacity": dh.plot_capacity_change_analysis(figsize=(8, 5)),
            "clipping": dh.plot_clipping(figsize=(12, 6)),
            "timeshift": tsh,
        }
        return out
    return make_static_plots,


@app.cell
def __(executed_siteid, executed_sysid, make_static_plots):
    static_plots = make_static_plots(executed_siteid, executed_sysid)
    return static_plots,


@app.cell
def __(mo):
    selection_switch = mo.ui.switch(label="system list / cache list")
    return selection_switch,


@app.cell
def __(mo):
    data_switch = mo.ui.switch(label="raw / cleaned data")
    return data_switch,


if __name__ == "__main__":
    app.run()
