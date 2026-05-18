from masfactory import Node, RootGraph
from typing import TypeVar
from masfactory.core.gate import Gate
from masfactory.utils.hook import masf_hook, HookStage

ConditionType = TypeVar("ConditionType")

class InputParser(Node):
    """
    Node responsible for parsing the input data and routing it to the appropriate nodes based on the content of the message:
    - routes the image data to the VisionAgentNode - **very important to use this specific node name for the vision agent** 
    - routes the rest of the data to other nodes
    """

    class Hook:
        EXECUTE = HookStage('execute')
        FORWARD = HookStage('forward')
        BUILD = HookStage('build')
        MESSAGE_AGGREGATE_IN = HookStage('message_aggregate_in')
        MESSAGE_DISPATCH_OUT = HookStage('message_dispatch_out')

    def __init__(self,
            name:str,
            pull_keys:dict[str,dict|str]|None=None,
            push_keys:dict[str,dict|str]|None=None,
            attributes:dict[str,object] | None = None,
        ):
            super().__init__(
                name=name,
                pull_keys=pull_keys,
                push_keys=push_keys,
                attributes=attributes
            )
    
    @masf_hook(Hook.FORWARD)
    def _forward(self, input:dict[str,object]) -> dict[str,object]:
        """
        Args:
            input (dict[str,object]): Node input payload.
        Returns:
            dict[str,object]: Node output payload.
        """
        return input
    
    @masf_hook(Hook.MESSAGE_DISPATCH_OUT)
    def _message_dispatch_out(self, message):
        """
        Modified message dispatching logic to route messages based on the destination node type.

        Dispatch the input to the right nodes (through the right edges):
        - if the edge is towards the VisionAgentNode, send only the image data (message["image"])
        - if the edge is towards the ConditionalNode, send only the location data (message["location"])
        - if the edge is towards other nodes, send the whole message but the image
        """

        # Message to be sent to non-vision nodes
        message_without_image = {k: v for k, v in message.items() if k != "image"}

        for out_edge in self.out_edges:
            if out_edge._receiver._name == "VisionAgentNode":
                # Send only the image data to the VisionAgentNode
                out_edge.send_message({"image": message["image"]})
            if out_edge._receiver._name == "ConditionalNode":
                # Send only the location data to the ConditionalNode
                out_edge.send_message({"location": message["location"]})
            if out_edge._receiver._name != "VisionAgentNode" and out_edge._receiver._name != "ConditionalNode":
                out_edge.send_message(message_without_image)

        self._gate = Gate.OPEN

def create_input_parser_node(graph: RootGraph, node_name: str = "InputParserNode"):
    return graph.create_node(
        InputParser,
        name=node_name
    )