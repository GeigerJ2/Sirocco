import pytest
from pathlib import Path
from aiida import orm
import textwrap

from sirocco.core import Workflow, AvailableData, GeneratedData
from sirocco.parsing import yaml_data_models as models
from sirocco.workgraph import AiidaWorkGraph


def test_get_aiida_label_from_graph_item(tmp_path):
    """Test that AiiDA labels are generated correctly."""

    # Mock data nodes with different coordinate combinations
    output_path = tmp_path / "output"
    data_simple = GeneratedData(
        name="output", type=models.DataType.FILE, src=output_path, coordinates={}
    )

    data_with_date = GeneratedData(
        name="output",
        type=models.DataType.FILE,
        src=output_path,
        coordinates={"date": "2026-01-01-00:00:00"},
    )

    data_with_params = GeneratedData(
        name="output",
        type=models.DataType.FILE,
        src=output_path,
        coordinates={"foo": 0, "bar": 3.0, "date": "2026-01-01-00:00:00"},
    )

    # Test label generation
    assert AiidaWorkGraph.get_aiida_label_from_graph_item(data_simple) == "output"
    assert (
        AiidaWorkGraph.get_aiida_label_from_graph_item(data_with_date)
        == "output_date_2026_01_01_00_00_00"
    )
    assert (
        AiidaWorkGraph.get_aiida_label_from_graph_item(data_with_params)
        == "output_foo_0___bar_3_0___date_2026_01_01_00_00_00"
    )


def test_filename_conflict_detection(tmp_path):
    """Test logic for detecting when unique filenames are needed."""

    output_path = tmp_path / "output"
    other_path = tmp_path / "other"

    inputs = [
        GeneratedData(
            name="output",
            type=models.DataType.FILE,
            src=output_path,
            coordinates={"foo": 0},
        ),
        GeneratedData(
            name="output",
            type=models.DataType.FILE,
            src=output_path,
            coordinates={"foo": 1},
        ),
        GeneratedData(
            name="other",
            type=models.DataType.FILE,
            src=other_path,
            coordinates={},
        ),
    ]

    # Test that conflict detection works
    output_conflicts = [inp for inp in inputs if inp.name == "output"]
    other_conflicts = [inp for inp in inputs if inp.name == "other"]

    assert len(output_conflicts) == 2  # Should need unique filenames
    assert len(other_conflicts) == 1  # Should use simple filename



@pytest.mark.usefixtures('aiida_localhost')
def test_basic_remote_data_filename(tmp_path):
    """Test basic RemoteData filename handling."""
    file_path = tmp_path / "foo.txt"
    file_path.touch()
    script_path = tmp_path / "script.sh"
    script_path.touch()

    config_wf = models.ConfigWorkflow(
        name="basic",
        rootdir=tmp_path,
        cycles=[
            models.ConfigCycle(
                name="cycle",
                tasks=[
                    models.ConfigCycleTask(
                        name="task",
                        inputs=[
                            models.ConfigCycleTaskInput(name="data", port="input")
                        ],
                    )
                ],
            ),
        ],
        tasks=[
            models.ConfigShellTask(
                name="task",
                command="echo {PORT::input}",
                src=str(script_path),
                computer="localhost",
            ),
        ],
        data=models.ConfigData(
            available=[
                models.ConfigAvailableData(
                    name="data",
                    type=models.DataType.FILE,
                    src=str(file_path),
                    computer="localhost",
                )
            ],
        ),
    )

    core_wf = Workflow.from_config_workflow(config_wf)
    aiida_wf = AiidaWorkGraph(core_wf)

    # Check that RemoteData was created and filename is correct
    task = aiida_wf._workgraph.tasks[0]
    assert isinstance(task.inputs.nodes["data"].value, orm.RemoteData)
    assert task.inputs.filenames.value == {"data": "foo.txt"}
    assert task.inputs.arguments.value == "{data}"

@pytest.mark.usefixtures('aiida_localhost')
def test_parameterized_filename_conflicts(tmp_path):
    """Test that parameterized data gets unique filenames when conflicts occur."""
    yaml_content = textwrap.dedent("""
        name: test_workflow
        cycles:
            - simulation_cycle:
                tasks:
                    - simulate:
                        inputs:
                            - input_file:
                                port: input
                        outputs: [simulation_output]
            - processing_cycle:
                tasks:
                    - process_data:
                        inputs:
                            - simulation_output:
                                parameters:
                                    param: all
                                port: files
                        outputs: [processed_output]
        tasks:
            - simulate:
                plugin: shell
                command: "simulate.py {PORT::input}"
                parameters: [param]
                computer: localhost
            - process_data:
                plugin: shell
                command: "process.py {PORT::files}"
                parameters: [param]
                computer: localhost
        data:
            available:
                - input_file:
                    type: file
                    src: input.txt
                    computer: localhost
            generated:
                - simulation_output:
                    type: file
                    src: output.dat
                    parameters: [param]
                - processed_output:
                    type: file
                    src: processed.dat
                    parameters: [param]
        parameters:
            param: [1, 2]
        """)

    config_file = tmp_path / "config.yml"
    config_file.write_text(yaml_content)

    # Create required files
    (tmp_path / "input.txt").touch()
    (tmp_path / "process.py").touch()

    core_wf = Workflow.from_config_file(str(config_file))
    aiida_wf = AiidaWorkGraph(core_wf)

    # Find the task that processes multiple parameterized inputs
    process_tasks = [
        task
        for task in aiida_wf._workgraph.tasks
        if task.name.startswith("process_data")
    ]

    # Should have 2 tasks (one for each parameter value)
    assert len(process_tasks) == 2

    # Check that each task has unique node keys and appropriate filenames
    for task in process_tasks:
        nodes_keys = list(task.inputs.nodes._sockets.keys())
        filenames = task.inputs.filenames.value
        arguments = task.inputs.arguments.value

        # Each task should have exactly one simulation_output input
        sim_output_keys = [
            k for k in nodes_keys if k.startswith("simulation_output")
        ]
        assert len(sim_output_keys) == 1

        # The filename should be the full label (since there are conflicts)
        key = sim_output_keys[0]
        assert filenames[key] == key  # Full label used as filename
        assert key in arguments  # Key appears in arguments

@pytest.mark.usefixtures('aiida_localhost')
def test_parameterized_filename_conflicts(tmp_path):
    """Test that parameterized data gets unique filenames when conflicts occur."""
    yaml_content = textwrap.dedent("""
        name: test_workflow
        cycles:
            - simulation_cycle:
                tasks:
                    - simulate:
                        inputs:
                            - input_file:
                                port: input
                        outputs: [simulation_output]
            - processing_cycle:
                tasks:
                    - process_data:
                        inputs:
                            - simulation_output:
                                parameters:
                                    param: all
                                port: files
                        outputs: [processed_output]
        tasks:
            - simulate:
                plugin: shell
                command: "/tmp/simulate.py {PORT::input}"
                parameters: [param]
                computer: localhost
            - process_data:
                plugin: shell
                command: "/tmp/process.py {PORT::files}"
                parameters: [param]
                computer: localhost
        data:
            available:
                - input_file:
                    type: file
                    src: input.txt
                    computer: localhost
            generated:
                - simulation_output:
                    type: file
                    src: output.dat
                    parameters: [param]
                - processed_output:
                    type: file
                    src: processed.dat
                    parameters: [param]
        parameters:
            param: [1, 2]
        """)

    config_file = tmp_path / "config.yml"
    config_file.write_text(yaml_content)

    # Create required files
    (tmp_path / "input.txt").touch()
    (tmp_path / "simulate.py").touch()
    (tmp_path / "process.py").touch()

    core_wf = Workflow.from_config_file(str(config_file))
    aiida_wf = AiidaWorkGraph(core_wf)

    # Find the task that processes multiple parameterized inputs
    process_tasks = [
        task
        for task in aiida_wf._workgraph.tasks
        if task.name.startswith("process_data")
    ]

    # Should have 2 tasks (one for each parameter value)
    assert len(process_tasks) == 2

    # Check that each task has unique node keys and appropriate filenames
    for task in process_tasks:
        nodes_keys = list(task.inputs.nodes._sockets.keys())
        filenames = task.inputs.filenames.value
        arguments = task.inputs.arguments.value

        # Each task should have two simulation outputs as input
        sim_output_keys = [
            k for k in nodes_keys if k.startswith("simulation_output")
        ]
        import ipdb; ipdb.set_trace()
        assert len(sim_output_keys) == 2

        # The filename should be the full label (since there are conflicts)
        key = sim_output_keys[0]
        assert filenames[key] == key  # Full label used as filename
        assert key in arguments  # Key appears in arguments

def test_mixed_conflict_and_no_conflict(tmp_path):
    """Test workflow with both conflicting and non-conflicting data."""
    yaml_content = textwrap.dedent("""
    cycles:
        - test_cycle:
            tasks:
            - analyze:
                inputs:
                    - shared_config:  # Single file, no conflict
                        port: config
                    - simulation_data:  # Multiple files, conflict expected
                        parameters:
                        run: all
                        port: data
                outputs: [analysis_result]
    tasks:
        - analyze:
            plugin: shell
            command: "analyze.py --config {PORT::config} --data {PORT::data}"
            src: analyze.py
            parameters: [run]
            computer: localhost
    data:
        available:
        - shared_config:
            type: file
            src: config.json
            computer: localhost
        generated:
        - simulation_data:
            type: file
            src: sim_output.nc
            parameters: [run]
        - analysis_result:
            type: file
            src: result.txt
            parameters: [run]
    parameters:
        run: [1, 2]
    """)

    config_file = tmp_path / "config.yml"
    config_file.write_text(yaml_content)

    # Create files
    (tmp_path / "config.json").touch()
    (tmp_path / "analyze.py").touch()

    core_wf = Workflow.from_config_file(str(config_file))
    aiida_wf = AiidaWorkGraph(core_wf)

    analyze_tasks = [
        task
        for task in aiida_wf._workgraph.tasks
        if task.name.startswith("analyze")
    ]

    for task in analyze_tasks:
        filenames = task.inputs.filenames.value

        # shared_config should use simple filename (no conflict)
        assert filenames["shared_config"] == "config.json"

        # simulation_data should use full label (conflict with other tasks)
        sim_data_key = [
            k for k in filenames.keys() if k.startswith("simulation_data")
        ][0]
        assert filenames[sim_data_key] == sim_data_key  # Full label as filename

@pytest.mark.usefixtures("aiida_localhost")
def test_comprehensive_parameterized_workflow(tmp_path):
    """Comprehensive test covering the full parameterized workflow scenario.

    This test validates the complete integration including:
    - Multiple parameters (foo, bar) and dates
    - Mixed conflict/no-conflict scenarios
    - Correct argument resolution
    - Proper filename mapping
    """
    yaml_str = textwrap.dedent("""
        start_date: &start "2026-01-01T00:00"
        stop_date: &stop "2026-07-01T00:00"
        cycles:
          - main:
              cycling:
                start_date: *start
                stop_date: *stop
                period: P6M
              tasks:
                - simulate:
                    inputs:
                      - config:
                          port: cfg
                    outputs: [sim_output]
                - analyze:
                    inputs:
                      - sim_output:
                          parameters: {foo: all, bar: single}
                          port: data
                    outputs: [analysis]
        tasks:
          - simulate:
              plugin: shell
              command: "sim.py {PORT::cfg}"
              parameters: [foo, bar]
              computer: localhost
          - analyze:
              plugin: shell
              command: "analyze.py {PORT::data}"
              parameters: [bar]
              computer: localhost
        data:
          available:
            - config:
                type: file
                src: config.txt
                computer: localhost
          generated:
            - sim_output:
                type: file
                src: output.dat
                parameters: [foo, bar]
            - analysis:
                type: file
                src: analysis.txt
                parameters: [bar]
        parameters:
          foo: [0, 1]
          bar: [3.0]
    """)

    config_file = tmp_path / "config.yml"
    config_file.write_text(yaml_str)

    # Create files
    (tmp_path / "config.txt").touch()
    (tmp_path / "sim.py").touch()
    (tmp_path / "analyze.py").touch()

    core_wf = Workflow.from_config_file(str(config_file))
    aiida_wf = AiidaWorkGraph(core_wf)

    # Verify task structure
    sim_tasks = [t for t in aiida_wf._workgraph.tasks if t.name.startswith("simulate")]
    analyze_tasks = [
        t for t in aiida_wf._workgraph.tasks if t.name.startswith("analyze")
    ]

    assert len(sim_tasks) == 2  # 2 foo values × 1 bar value = 2 tasks
    assert len(analyze_tasks) == 1  # 1 bar value = 1 task

    # Check simulate tasks (should have simple config filename)
    for task in sim_tasks:
        filenames = task.inputs.filenames.value
        assert filenames["config"] == "config.txt"  # No conflict, simple name

    # Check analyze task (should have complex filenames due to conflicts)
    analyze_task = analyze_tasks[0]
    filenames = analyze_task.inputs.filenames.value

    # Should have 2 sim_output inputs with full labels as filenames
    sim_output_keys = [k for k in filenames.keys() if k.startswith("sim_output")]
    assert len(sim_output_keys) == 2

    for key in sim_output_keys:
        assert filenames[key] == key  # Full label used as filename
        assert "foo_" in key and "bar_3_0" in key  # Contains parameter info


# @pytest.fixture
# def sample_workflow_config(tmp_path):
#     """Fixture providing a reusable workflow configuration for testing."""
#     config = {
#         "name": "test_workflow",
#         "rootdir": tmp_path,
#         "cycles": [],
#         "tasks": [],
#         "data": models.ConfigData(),
#         "parameters": {},
#     }
#     return config


# @pytest.mark.usefixtures("aiida_localhost")
# def test_linking_complex_dates_and_parameters(tmp_path):
#     yaml_str = textwrap.dedent(
#         """
#         start_date: &root_start_date "2026-01-01T00:00"
#         stop_date: &root_stop_date "2028-01-01T00:00"
#         cycles:
#             - bimonthly_tasks:
#                 cycling:
#                     start_date: *root_start_date
#                     stop_date: *root_stop_date
#                     period: P6M
#                 tasks:
#                     - icon:
#                         inputs:
#                             - initial_conditions:
#                                 when:
#                                     at: *root_start_date
#                                 port: init
#                             - icon_restart:
#                                 when:
#                                     after: *root_start_date
#                                 target_cycle:
#                                     lag: -P6M
#                                 parameters:
#                                     foo: single
#                                     bar: single
#                                 port: restart
#                             - forcing:
#                                 port: forcing
#                         outputs: [icon_output, icon_restart]
#                     - statistics_foo:
#                         inputs:
#                             - icon_output:
#                                 parameters:
#                                     bar: single
#                                 port: None
#                         outputs: [analysis_foo]
#                     - statistics_foo_bar:
#                         inputs:
#                             - analysis_foo:
#                                 port: None
#                         outputs: [analysis_foo_bar]
#             - yearly:
#                 cycling:
#                     start_date: *root_start_date
#                     stop_date: *root_stop_date
#                     period: P1Y
#                 tasks:
#                     - merge:
#                         inputs:
#                             - analysis_foo_bar:
#                                 target_cycle:
#                                     lag: ["P0M", "P6M"]
#                                 port: None
#                         outputs: [yearly_analysis]
#         tasks:
#             - icon:
#                 plugin: shell
#                 src: /home/geiger_j/aiida_projects/swiss-twins/git-repos/Sirocco/tests/cases/parameters/config/scripts/icon.py
#                 command: "icon.py --restart {PORT::restart} --init {PORT::init} --forcing {PORT::forcing}"
#                 parameters: [foo, bar]
#                 computer: localhost
#             - statistics_foo:
#                 plugin: shell
#                 src: /home/geiger_j/aiida_projects/swiss-twins/git-repos/Sirocco/tests/cases/parameters/config/scripts/statistics.py
#                 command: "statistics.py {PORT::None}"
#                 parameters: [bar]
#                 computer: localhost
#             - statistics_foo_bar:
#                 plugin: shell
#                 src: /home/geiger_j/aiida_projects/swiss-twins/git-repos/Sirocco/tests/cases/parameters/config/scripts/statistics.py
#                 command: "statistics.py {PORT::None}"
#                 computer: localhost
#             - merge:
#                 plugin: shell
#                 src: /home/geiger_j/aiida_projects/swiss-twins/git-repos/Sirocco/tests/cases/parameters/config/scripts/merge.py
#                 command: "merge.py {PORT::None}"
#                 computer: localhost
#         data:
#             available:
#                 - initial_conditions:
#                     type: file
#                     src: /home/geiger_j/aiida_projects/swiss-twins/git-repos/Sirocco/tests/cases/small/config/data/initial_conditions
#                     computer: localhost
#                 - forcing:
#                     type: file
#                     src: /home/geiger_j/aiida_projects/swiss-twins/git-repos/Sirocco/tests/cases/parameters/config/data/forcing
#                     computer: localhost
#             generated:
#                 - icon_output:
#                     type: file
#                     src: icon_output
#                     parameters: [foo, bar]
#                 - icon_restart:
#                     type: file
#                     src: restart
#                     parameters: [foo, bar]
#                 - analysis_foo:
#                     type: file
#                     src: analysis
#                     parameters: [bar]
#                 - analysis_foo_bar:
#                     type: file
#                     src: analysis
#                 - yearly_analysis:
#                     type: file
#                     src: analysis
#         parameters:
#             foo: [0, 1]
#             bar: [3.0]
#         """
#     )
#     yaml_file = tmp_path / "config.yml"
#     yaml_file.write_text(yaml_str)

#     core_wf = Workflow.from_config_file(yaml_file)
#     aiida_wf = AiidaWorkGraph(core_workflow=core_wf)
#     filenames_list = [task.inputs.filenames.value for task in aiida_wf._workgraph.tasks]
#     arguments_list = [task.inputs.arguments.value for task in aiida_wf._workgraph.tasks]
#     nodes_list = [
#         list(task.inputs.nodes._sockets.keys()) for task in aiida_wf._workgraph.tasks
#     ]

#     expected_filenames_list = [
#         {"forcing": "forcing", "initial_conditions": "initial_conditions"},
#         {"forcing": "forcing", "initial_conditions": "initial_conditions"},
#         {
#             "forcing": "forcing",
#             "icon_restart_foo_0___bar_3_0___date_2026_01_01_00_00_00": "restart",
#         },
#         {
#             "forcing": "forcing",
#             "icon_restart_foo_1___bar_3_0___date_2026_01_01_00_00_00": "restart",
#         },
#         {
#             "forcing": "forcing",
#             "icon_restart_foo_0___bar_3_0___date_2026_07_01_00_00_00": "restart",
#         },
#         {
#             "forcing": "forcing",
#             "icon_restart_foo_1___bar_3_0___date_2026_07_01_00_00_00": "restart",
#         },
#         {
#             "forcing": "forcing",
#             "icon_restart_foo_0___bar_3_0___date_2027_01_01_00_00_00": "restart",
#         },
#         {
#             "forcing": "forcing",
#             "icon_restart_foo_1___bar_3_0___date_2027_01_01_00_00_00": "restart",
#         },
#         {
#             "icon_output_foo_0___bar_3_0___date_2026_01_01_00_00_00": "icon_output_foo_0___bar_3_0___date_2026_01_01_00_00_00",
#             "icon_output_foo_1___bar_3_0___date_2026_01_01_00_00_00": "icon_output_foo_1___bar_3_0___date_2026_01_01_00_00_00",
#         },
#         {
#             "icon_output_foo_0___bar_3_0___date_2026_07_01_00_00_00": "icon_output_foo_0___bar_3_0___date_2026_07_01_00_00_00",
#             "icon_output_foo_1___bar_3_0___date_2026_07_01_00_00_00": "icon_output_foo_1___bar_3_0___date_2026_07_01_00_00_00",
#         },
#         {
#             "icon_output_foo_0___bar_3_0___date_2027_01_01_00_00_00": "icon_output_foo_0___bar_3_0___date_2027_01_01_00_00_00",
#             "icon_output_foo_1___bar_3_0___date_2027_01_01_00_00_00": "icon_output_foo_1___bar_3_0___date_2027_01_01_00_00_00",
#         },
#         {
#             "icon_output_foo_0___bar_3_0___date_2027_07_01_00_00_00": "icon_output_foo_0___bar_3_0___date_2027_07_01_00_00_00",
#             "icon_output_foo_1___bar_3_0___date_2027_07_01_00_00_00": "icon_output_foo_1___bar_3_0___date_2027_07_01_00_00_00",
#         },
#         {"analysis_foo_bar_3_0___date_2026_01_01_00_00_00": "analysis"},
#         {"analysis_foo_bar_3_0___date_2026_07_01_00_00_00": "analysis"},
#         {"analysis_foo_bar_3_0___date_2027_01_01_00_00_00": "analysis"},
#         {"analysis_foo_bar_3_0___date_2027_07_01_00_00_00": "analysis"},
#         {
#             "analysis_foo_bar_date_2026_01_01_00_00_00": "analysis_foo_bar_date_2026_01_01_00_00_00",
#             "analysis_foo_bar_date_2026_07_01_00_00_00": "analysis_foo_bar_date_2026_07_01_00_00_00",
#         },
#         {
#             "analysis_foo_bar_date_2027_01_01_00_00_00": "analysis_foo_bar_date_2027_01_01_00_00_00",
#             "analysis_foo_bar_date_2027_07_01_00_00_00": "analysis_foo_bar_date_2027_07_01_00_00_00",
#         },
#     ]

#     expected_arguments_list = [
#         "--restart  --init {initial_conditions} --forcing {forcing}",
#         "--restart  --init {initial_conditions} --forcing {forcing}",
#         "--restart {icon_restart_foo_0___bar_3_0___date_2026_01_01_00_00_00} --init  "
#         "--forcing {forcing}",
#         "--restart {icon_restart_foo_1___bar_3_0___date_2026_01_01_00_00_00} --init  "
#         "--forcing {forcing}",
#         "--restart {icon_restart_foo_0___bar_3_0___date_2026_07_01_00_00_00} --init  "
#         "--forcing {forcing}",
#         "--restart {icon_restart_foo_1___bar_3_0___date_2026_07_01_00_00_00} --init  "
#         "--forcing {forcing}",
#         "--restart {icon_restart_foo_0___bar_3_0___date_2027_01_01_00_00_00} --init  "
#         "--forcing {forcing}",
#         "--restart {icon_restart_foo_1___bar_3_0___date_2027_01_01_00_00_00} --init  "
#         "--forcing {forcing}",
#         "{icon_output_foo_0___bar_3_0___date_2026_01_01_00_00_00} "
#         "{icon_output_foo_1___bar_3_0___date_2026_01_01_00_00_00}",
#         "{icon_output_foo_0___bar_3_0___date_2026_07_01_00_00_00} "
#         "{icon_output_foo_1___bar_3_0___date_2026_07_01_00_00_00}",
#         "{icon_output_foo_0___bar_3_0___date_2027_01_01_00_00_00} "
#         "{icon_output_foo_1___bar_3_0___date_2027_01_01_00_00_00}",
#         "{icon_output_foo_0___bar_3_0___date_2027_07_01_00_00_00} "
#         "{icon_output_foo_1___bar_3_0___date_2027_07_01_00_00_00}",
#         "{analysis_foo_bar_3_0___date_2026_01_01_00_00_00}",
#         "{analysis_foo_bar_3_0___date_2026_07_01_00_00_00}",
#         "{analysis_foo_bar_3_0___date_2027_01_01_00_00_00}",
#         "{analysis_foo_bar_3_0___date_2027_07_01_00_00_00}",
#         "{analysis_foo_bar_date_2026_01_01_00_00_00} "
#         "{analysis_foo_bar_date_2026_07_01_00_00_00}",
#         "{analysis_foo_bar_date_2027_01_01_00_00_00} "
#         "{analysis_foo_bar_date_2027_07_01_00_00_00}",
#     ]

#     expected_nodes_list = [
#         ["initial_conditions", "forcing"],
#         ["initial_conditions", "forcing"],
#         ["icon_restart_foo_0___bar_3_0___date_2026_01_01_00_00_00", "forcing"],
#         ["icon_restart_foo_1___bar_3_0___date_2026_01_01_00_00_00", "forcing"],
#         ["icon_restart_foo_0___bar_3_0___date_2026_07_01_00_00_00", "forcing"],
#         ["icon_restart_foo_1___bar_3_0___date_2026_07_01_00_00_00", "forcing"],
#         ["icon_restart_foo_0___bar_3_0___date_2027_01_01_00_00_00", "forcing"],
#         ["icon_restart_foo_1___bar_3_0___date_2027_01_01_00_00_00", "forcing"],
#         [
#             "icon_output_foo_0___bar_3_0___date_2026_01_01_00_00_00",
#             "icon_output_foo_1___bar_3_0___date_2026_01_01_00_00_00",
#         ],
#         [
#             "icon_output_foo_0___bar_3_0___date_2026_07_01_00_00_00",
#             "icon_output_foo_1___bar_3_0___date_2026_07_01_00_00_00",
#         ],
#         [
#             "icon_output_foo_0___bar_3_0___date_2027_01_01_00_00_00",
#             "icon_output_foo_1___bar_3_0___date_2027_01_01_00_00_00",
#         ],
#         [
#             "icon_output_foo_0___bar_3_0___date_2027_07_01_00_00_00",
#             "icon_output_foo_1___bar_3_0___date_2027_07_01_00_00_00",
#         ],
#         ["analysis_foo_bar_3_0___date_2026_01_01_00_00_00"],
#         ["analysis_foo_bar_3_0___date_2026_07_01_00_00_00"],
#         ["analysis_foo_bar_3_0___date_2027_01_01_00_00_00"],
#         ["analysis_foo_bar_3_0___date_2027_07_01_00_00_00"],
#         [
#             "analysis_foo_bar_date_2026_01_01_00_00_00",
#             "analysis_foo_bar_date_2026_07_01_00_00_00",
#         ],
#         [
#             "analysis_foo_bar_date_2027_01_01_00_00_00",
#             "analysis_foo_bar_date_2027_07_01_00_00_00",
#         ],
#     ]

#     assert arguments_list == expected_arguments_list
#     assert filenames_list == expected_filenames_list
#     assert nodes_list == expected_nodes_list
