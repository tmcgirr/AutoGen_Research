import os
from getpass import getpass
import asyncio
import platform
from typing import List, Optional, TypedDict
import base64
from io import BytesIO
from IPython.display import HTML, display
from PIL import Image
import re
import time

import playwright
from IPython import display
from playwright.async_api import async_playwright

from langchain_core.runnables import chain as chain_decorator
from langchain_core.runnables import RunnableLambda
from langgraph.graph import END, StateGraph
from langchain_core.messages import BaseMessage, SystemMessage
from playwright.async_api import Page

from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

from langchain.prompts.chat import ChatPromptTemplate
from langchain.prompts.chat import SystemMessagePromptTemplate
from langchain.prompts.chat import MessagesPlaceholder
from langchain.prompts.chat import HumanMessagePromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts.image import ImagePromptTemplate

# with open("mark_page.js") as f:
with open("Tools/mark_page.js") as f:
    mark_page_script = f.read()

# Import env variables (load from .env file)
from dotenv import load_dotenv
load_dotenv()

# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_PROJECT"] = "Web-Voyager"
# _getpass("LANGCHAIN_API_KEY")

# Load the Llama model
# from langchain_community.llms import Ollama
# llava_model = Ollama(model="llava:latest")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_ORG = os.getenv("OPENAI_ORG")


################################################
# Set Graph State
################################################
class BBox(TypedDict):
    x: float
    y: float
    text: str
    type: str
    ariaLabel: str

class Prediction(TypedDict):
    action: str
    args: Optional[List[str]]

# This represents the state of the agent as it proceeds through execution
class AgentState(TypedDict):
    page: Page  # The Playwright web page lets us interact with the web environment
    input: str  # User request
    img: str  # b64 encoded screenshot
    bboxes: List[BBox]  # The bounding boxes from the browser annotation function
    prediction: Prediction  # The Agent's output
    # A system message (or messages) containing the intermediate steps
    scratchpad: List[BaseMessage]
    observation: str  # The most recent response from a tool
    step: int  # The current step in the agent's execution
    
################################################
# Agent Tools
################################################
async def click(state: AgentState):
    # - Click [Numerical_Label]
    page = state["page"]
    click_args = state["prediction"]["args"]
    if click_args is None or len(click_args) != 1:
        return f"Failed to click bounding box labeled as number {click_args}"
    bbox_id = click_args[0]
    bbox_id = int(bbox_id)
    try:
        bbox = state["bboxes"][bbox_id]
    except:
        return f"Error: no bbox for : {bbox_id}"
    x, y = bbox["x"], bbox["y"]
    res = await page.mouse.click(x, y)
    # TODO: In the paper, they automatically parse any downloaded PDFs
    # We could add something similar here as well and generally
    # improve response format.
    return f"Clicked {bbox_id}"


async def type_text(state: AgentState):
    page = state["page"]
    type_args = state["prediction"]["args"]
    if type_args is None or len(type_args) != 2:
        return (
            f"Failed to type in element from bounding box labeled as number {type_args}"
        )
    bbox_id_str, text_content = type_args
    try:
        bbox_id = int(bbox_id_str)
    except ValueError:
        return f"Error: invalid bounding box ID '{bbox_id_str}'"
    
    # Check if bbox_id is within the valid range
    if bbox_id < 0 or bbox_id >= len(state["bboxes"]):
        return f"Error: bbox ID {bbox_id} is out of range"
    
    bbox = state["bboxes"][bbox_id]
    x, y = bbox["x"], bbox["y"]
    await page.mouse.click(x, y)
    # Check if MacOS
    select_all = "Meta+A" if platform.system() == "Darwin" else "Control+A"
    await page.keyboard.press(select_all)
    await page.keyboard.press("Backspace")
    await page.keyboard.type(text_content)
    await page.keyboard.press("Enter")
    return f"Typed {text_content} and submitted"


async def scroll(state: AgentState):
    page = state["page"]
    scroll_args = state["prediction"]["args"]
    if scroll_args is None or len(scroll_args) != 2:
        return "Failed to scroll due to incorrect arguments."

    target, direction = scroll_args

    if target.upper() == "WINDOW":
        # Not sure the best value for this:
        scroll_amount = 500
        scroll_direction = (
            -scroll_amount if direction.lower() == "up" else scroll_amount
        )
        await page.evaluate(f"window.scrollBy(0, {scroll_direction})")
    else:
        # Scrolling within a specific element
        scroll_amount = 200
        target_id = int(target)
        bbox = state["bboxes"][target_id]
        x, y = bbox["x"], bbox["y"]
        scroll_direction = (
            -scroll_amount if direction.lower() == "up" else scroll_amount
        )
        await page.mouse.move(x, y)
        await page.mouse.wheel(0, scroll_direction)

    return f"Scrolled {direction} in {'window' if target.upper() == 'WINDOW' else 'element'}"

async def wait(state: AgentState):
    sleep_time = 5
    await asyncio.sleep(sleep_time)
    return f"Waited for {sleep_time}s."

async def go_back(state: AgentState):
    page = state["page"]
    await page.go_back()
    return f"Navigated back a page to {page.url}."

async def to_google(state: AgentState):
    page = state["page"]
    await page.goto("https://www.google.com/")
    return "Navigated to google.com."

# Agent Tool (Scratchpad)
def update_scratchpad(state: AgentState):
    """After a tool is invoked, we want to update
    the scratchpad so the agent is aware of its previous steps"""
    old = state.get("scratchpad")
    if old:
        txt = old[0].content
        last_line = txt.rsplit("\n", 1)[-1]
        step = int(re.match(r"\d+", last_line).group()) + 1
    else:
        txt = "Previous action observations:\n"
        step = 1
    txt += f"\n{step}. {state['observation']}"
    
    # Create or append to the Scratchpad file (for visibility)
     # Ensure the directory exists before writing to the file
    scratchpad_path = "./AgentNotebook/agent_scratchpad.txt"
    os.makedirs(os.path.dirname(scratchpad_path), exist_ok=True)

    # Create or append to the Scratchpad file (for visibility)
    with open(scratchpad_path, "a+") as file:
        file.write(f"{step}. {state['observation']}\n")

    return {**state, "scratchpad": [SystemMessage(content=txt)]}
  
# Required for the agent to process the image (Required to reduce size for local llm)
# def process_and_encode_image(image_bytes):
#     # Convert bytes data to a file-like object
#     image_file = io.BytesIO(image_bytes)

#     # Open the image file
#     with Image.open(image_file) as img:
#         # Convert RGBA to RGB if necessary
#         if img.mode == 'RGBA':
#             img = img.convert('RGB')

#         # Resize the image to half of its original size
#         img = img.resize((img.width // 2, img.height // 2))
        
#         # Convert image to JPEG format with reduced quality
#         img_byte_arr = io.BytesIO()
#         img.save(img_byte_arr, format='JPEG', quality=75)  # Adjust quality for further size reduction
        
#         # Encode to base64
#         encoded_image = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
        
#     return encoded_image

# Browser Annotation
@chain_decorator
async def mark_page(page):
    await page.evaluate(mark_page_script)
    for _ in range(10):
        try:
            bboxes = await page.evaluate("markPage()")
            break
        except:
            # Properly awaiting the sleep to ensure the delay is effective
            await asyncio.sleep(3)
    screenshot = await page.screenshot()
    
    # save each screenshot to a directory
    screenshot_path = "./AgentNotebook/screenshots/agent_screenshot" + time.strftime("%Y%m%d-%H%M%S") + ".png"
    os.makedirs(os.path.dirname(screenshot_path), exist_ok=True)
    with open(screenshot_path, "wb") as file:
        file.write(screenshot)
        
    # Return the screenshot and bounding boxes
    await page.evaluate("unmarkPage()")
    return {
        "img": base64.b64encode(screenshot).decode(),
        # "img": process_and_encode_image(screenshot), # Required for local llm
        "bboxes": bboxes,
    }

################################################
# Agent Definition
async def annotate(state):
    marked_page = await mark_page.with_retry().ainvoke(state["page"])
    return {**state, **marked_page}

def format_descriptions(state):
    labels = []
    for i, bbox in enumerate(state["bboxes"]):
        text = bbox.get("ariaLabel") or ""
        if not text.strip():
            text = bbox["text"]
        el_type = bbox.get("type")
        labels.append(f'{i} (<{el_type}/>): "{text}"')
    bbox_descriptions = "\nValid Bounding Boxes:\n" + "\n".join(labels)
    return {**state, "bbox_descriptions": bbox_descriptions}


def parse(text: str) -> dict:
    action_prefix = "Action: "
    if not text.strip().split("\n")[-1].startswith(action_prefix):
        return {"action": "retry", "args": f"Could not parse LLM Output: {text}"}
    action_block = text.strip().split("\n")[-1]

    action_str = action_block[len(action_prefix) :]
    split_output = action_str.split(" ", 1)
    if len(split_output) == 1:
        action, action_input = split_output[0], None
    else:
        action, action_input = split_output
    action = action.strip()
    if action_input is not None:
        action_input = [
            inp.strip().strip("[]") for inp in action_input.strip().split(";")
        ]
    return {"action": action, "args": action_input}


# prompt = hub.pull("wfh/web-voyager")
prompt = ChatPromptTemplate(
  messages=[
    SystemMessagePromptTemplate(
      prompt=[
        PromptTemplate.from_template("Imagine you are a robot browsing the web, just like humans. Now you need to complete a task. In each iteration, you will receive an Observation that includes a screenshot of a webpage and some texts. This screenshot will\nfeature Numerical Labels placed in the TOP LEFT corner of each Web Element. Carefully analyze the visual\ninformation to identify the Numerical Label corresponding to the Web Element that requires interaction, then follow\nthe guidelines and choose one of the following actions:\n\n1. Click a Web Element.\n2. Delete existing content in a textbox and then type content.\n3. Scroll up or down.\n4. Wait \n5. Go back\n7. Return to google to start over.\n8. Webscrape the current page to save the content\n9. Respond with the final answer\n\nCorrespondingly, Action should STRICTLY follow the format:\n\n- Click [Numerical_Label] \n- Type [Numerical_Label]; [Content] \n- Scroll [Numerical_Label or WINDOW]; [up or down] \n- Wait \n- GoBack\n- Google\n- ANSWER; [content]\n\nKey Guidelines You MUST follow:\n\n* Action guidelines *\n1) Execute only one action per iteration.\n2) When clicking or typing, ensure to select the correct bounding box.\n3) Numeric labels lie in the top-left corner of their corresponding bounding boxes and are colored the same.\n\n* Web Browsing Guidelines *\n1) Don't interact with useless web elements like Login, Sign-in, donation that appear in Webpages\n2) Select strategically to minimize time wasted.\n\nYour reply should strictly follow the format:\n\nThought: {{Your brief thoughts (briefly summarize the info that will help ANSWER)}}\nAction: {{One Action format you choose}}\nThen the User will provide:\nObservation: {{A labeled screenshot Given by User}}\n"),
      ],
    ),
    MessagesPlaceholder(
      optional=True,
      variable_name="scratchpad",
    ),
    HumanMessagePromptTemplate(
      prompt=[
        ImagePromptTemplate(
          template={"url":"data:image/png;base64,{img}"},
          input_variables=[
            "img",
          ],
        ),
        PromptTemplate.from_template("{bbox_descriptions}"),
        PromptTemplate.from_template("{input}"),
      ],
    ),
  ],
  input_variables=[
    "bbox_descriptions",
    "img",
    "input",
  ],
  partial_variables={"scratchpad":[]},
)

################################################
# Define Language Model and Chain
llm = ChatOpenAI(model="gpt-4-vision-preview", temperature=0, api_key=OPENAI_API_KEY, openai_organization=OPENAI_ORG)

agent = annotate | RunnablePassthrough.assign(
    prediction=format_descriptions | prompt | llm | StrOutputParser() | parse
)

################################################
# Agent State Graph and Tool Integration
graph_builder = StateGraph(AgentState)
graph_builder.add_node("agent", agent)
graph_builder.set_entry_point("agent")
graph_builder.add_node("update_scratchpad", update_scratchpad)
graph_builder.add_edge("update_scratchpad", "agent")

tools = {
    "Click": click,
    "Type": type_text,
    "Scroll": scroll,
    "Wait": wait,
    "GoBack": go_back,
    "Google": to_google,
}

for node_name, tool in tools.items():
    graph_builder.add_node(
        node_name,
        # The lambda ensures the function's string output is mapped to the "observation"
        # key in the AgentState
        RunnableLambda(tool) | (lambda observation: {"observation": observation}),
    )
    # Always return to the agent (by means of the update-scratchpad node)
    graph_builder.add_edge(node_name, "update_scratchpad")

def select_tool(state: AgentState):
    action = state["prediction"]["action"]
    if action == "ANSWER":
        # print("Received ANSWER action, terminating graph.")  # Debug print
        return state["prediction"]["args"][0] if state["prediction"]["args"] else None
    if action == "retry":
        return "agent"
    return action

graph_builder.add_conditional_edges("agent", select_tool)
graph = graph_builder.compile()

# Tools for running the agent in a browser
async def setup_sandbox_browser():
    browser = await async_playwright().start()
    browser = await browser.chromium.launch(headless=False, args=None)
    page = await browser.new_page()
    _ = await page.goto("https://www.google.com")
    return page

# async def call_agent(question: str, page, max_steps: int = 150):
async def call_agent(question: str, page, max_steps: int = 500):
    event_stream = graph.astream(
        {
            "page": page,
            "input": question,
            "scratchpad": [],
        },
        {
            "recursion_limit": max_steps,
        },
    )
    steps = []
    async for event in event_stream:
        try:
            if "agent" not in event:
                continue
            pred = event["agent"].get("prediction") or {}
            action = pred.get("action")
            action_input = pred.get("args")
            display.clear_output(wait=False)
            step = f"{len(steps) + 1}. {action}: {action_input}"
            steps.append(step)
            print(step)  # Print each step
            
            # save all agent data to a file (entire event) but abbreviate image data
            agent_data_path = "./AgentNotebook/agent_data.txt"
            os.makedirs(os.path.dirname(agent_data_path), exist_ok=True)
            event_copy = dict(event)  # Create a copy of the event to modify
            if 'img' in event_copy['agent']:
                event_copy['agent']['img'] = '...'  # Abbreviate the image data
            with open(agent_data_path, "a+") as file:
                file.write(f"{event_copy}\n")
                
            action_messages = {
                "ANSWER": None,
                "retry": "Retrying...",
                "Google": "Navigating to Google...",
                "GoBack": "Navigating back...",
                "Wait": "Waiting...",
                "Scroll": "Scrolling...",
                "Type": "Typing...",
                "Click": "Clicking..."
            }

            if action in action_messages:
                message = action_messages[action]
                if message:
                    print(message)
                if action == "ANSWER":
                    break
                continue

            # If the action is not recognized, break the loop
            print("Action not recognized.")
            break
        
        except Exception as e:
            print(f"Error processing event: {e}")
            continue
    print(steps) 
    
    # Save steps to a new file
    # Ensure the directory exists before writing to the file
    steps_path = "./AgentNotebook/agent_steps.txt"
    os.makedirs(os.path.dirname(steps_path), exist_ok=True)
    with open(steps_path, "w") as file:
        for step in steps:
            file.write(f"{step}\n")
          
    return steps

################################################
# Run the agent 
async def execute_query(query: str):
    page = None  # Initialize page to None    
    try:
        page = await setup_sandbox_browser()
        res = await call_agent(
            query,
            page,
        )
        print(f"Final response: {res[-1]}")
        # Return the final response from the agent
        return res[-1]
    finally:
        if page:  # Check if page is not None
            await page.close()
            print("Browser closed.")

# Example query
# query = "Find the best restaurants in New York City."
# asyncio.run(execute_query(query))
