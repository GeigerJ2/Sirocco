from aiida import load_profile

from sirocco.core import Workflow
from sirocco.workgraph import AiidaWorkGraph

load_profile()


core_workflow = Workflow.from_config_file(
    "/home/geiger_j/aiida_projects/swiss-twins/git-repos/Sirocco/tests/cases/small/config/config.yml"
)
aiida_workflow = AiidaWorkGraph(core_workflow)
output_node = aiida_workflow.run()