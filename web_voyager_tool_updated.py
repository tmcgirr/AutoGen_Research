import os
import asyncio
import platform
from typing import List, Optional, TypedDict
import base64
import time
import json
from PIL import Image
import re
from openai import OpenAI

import playwright
from playwright.async_api import async_playwright, Page


# Import env variables
from dotenv import load_dotenv
load_dotenv()

# Setup clients based on model type
import ollama
from openai import OpenAI

# Model Configuration
# MODEL_TYPE = "ollama"
MODEL_TYPE = "openai"

# Ollama is close to working but not quite there yet
if MODEL_TYPE == "ollama":
    VISION_MODEL = os.getenv("VISION_MODEL", "llama3.2-vision:11b")
    client = ollama.AsyncClient()
else:
    VISION_MODEL = os.getenv("VISION_MODEL", "gpt-4o")
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        organization=os.getenv("OPENAI_ORG")
    )

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

class AgentState(TypedDict):
    page: Page
    input: str
    img: str
    bboxes: List[BBox]
    prediction: Prediction
    observation: str
    step: int
    visited_urls: List[str]
    action_count: dict

# Load the mark_page script
with open("Tools/WebVoyager/mark_page.js") as f:
    mark_page_script = f.read()

################################################
# Agent Tools
################################################
async def click(state: AgentState):
    page = state["page"]
    click_args = state["prediction"]["args"]
    if click_args is None or len(click_args) != 1:
        return f"Failed to click bounding box labeled as number {click_args}"
    bbox_id_str = click_args[0]
    # Remove any trailing periods or non-numeric characters
    bbox_id_str = ''.join(c for c in bbox_id_str if c.isdigit())
    try:
        bbox_id = int(bbox_id_str)
    except ValueError:
        return f"Error: invalid bbox ID '{bbox_id_str}'"
    try:
        bbox = state["bboxes"][bbox_id]
    except:
        return f"Error: no bbox for : {bbox_id}"
    x, y = bbox["x"], bbox["y"]
    await page.mouse.click(x, y)
    return f"Clicked {bbox_id}"

async def type_text(state: AgentState):
    page = state["page"]
    type_args = state["prediction"]["args"]
    if type_args is None or len(type_args) != 2:
        return f"Failed to type in element from bounding box labeled as number {type_args}"
    bbox_id_str, text_content = type_args
    try:
        bbox_id = int(bbox_id_str)
    except ValueError:
        return f"Error: invalid bounding box ID '{bbox_id_str}'"
    
    if bbox_id < 0 or bbox_id >= len(state["bboxes"]):
        return f"Error: bbox ID {bbox_id} is out of range"
    
    bbox = state["bboxes"][bbox_id]
    x, y = bbox["x"], bbox["y"]
    await page.mouse.click(x, y)
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
    scroll_amount = 500 if target.upper() == "WINDOW" else 200
    scroll_direction = -scroll_amount if direction.lower() == "up" else scroll_amount

    if target.upper() == "WINDOW":
        await page.evaluate(f"window.scrollBy(0, {scroll_direction})")
    else:
        target_id = int(target)
        bbox = state["bboxes"][target_id]
        x, y = bbox["x"], bbox["y"]
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

################################################
# Browser Annotation
################################################
async def mark_page(page: Page):
    await page.evaluate(mark_page_script)
    for _ in range(10):
        try:
            bboxes = await page.evaluate("markPage()")
            break
        except:
            await asyncio.sleep(3)
    screenshot = await page.screenshot()
    
    screenshot_path = "./Tools/WebVoyager/AgentNotebook/screenshots/agent_screenshot" + time.strftime("%Y%m%d-%H%M%S") + ".png"
    os.makedirs(os.path.dirname(screenshot_path), exist_ok=True)
    with open(screenshot_path, "wb") as file:
        file.write(screenshot)
        
    await page.evaluate("unmarkPage()")
    return {
        "img": base64.b64encode(screenshot).decode(),
        "bboxes": bboxes,
    }

################################################
# Agent Logic
################################################
def format_descriptions(bboxes: List[BBox]) -> str:
    labels = []
    for i, bbox in enumerate(bboxes):
        text = bbox.get("ariaLabel") or ""
        if not text.strip():
            text = bbox["text"]
        el_type = bbox.get("type")
        labels.append(f'{i} (<{el_type}/>): "{text}"')
    return "\nValid Bounding Boxes:\n" + "\n".join(labels)

def parse_action(text: str) -> dict:
    try:
        # Find the last line that contains "Action:"
        lines = text.strip().split('\n')
        action_line = None
        for line in reversed(lines):
            if "Action:" in line:
                action_line = line
                break
        
        if not action_line:
            return {"action": "retry", "args": f"Could not find Action: in output: {text}"}
            
        # Extract everything after "Action:"
        action_str = action_line.split("Action:", 1)[1].strip()
        
        # Remove any leading dash or bullet point
        action_str = action_str.lstrip("- ")
        
        # Split into action and arguments
        parts = action_str.split(" ", 1)
        action = parts[0].strip().rstrip(';')  # Remove any trailing semicolon
        
        # Handle arguments
        if len(parts) > 1:
            # Split arguments by semicolon, handling both space and no-space cases
            args_str = parts[1].strip()
            args = []
            for arg in args_str.replace('; ', ';').split(';'):
                # Clean up the argument
                arg = arg.strip().strip('[]').strip()
                # Remove any trailing periods
                if arg.endswith('.'):
                    arg = arg[:-1]
                args.append(arg)
        else:
            args = None
            
        print(f"Parsed action: {action}, args: {args}")  # Debug output
        return {"action": action, "args": args}
        
    except Exception as e:
        print(f"Error parsing action: {e}")
        return {"action": "retry", "args": f"Parse error: {str(e)}"}

async def get_agent_action(client, state: AgentState) -> Prediction:
    system_prompt = """
<task>
You are a robot browsing the web, just like humans. You will receive an Observation that includes a screenshot of a webpage and some texts. The screenshot contains Numerical Labels placed in the TOP LEFT corner of each Web Element.
</task>

<objective>
Carefully analyze the visual information to identify the Numerical Label corresponding to the Web Element that requires interaction. Choose exactly ONE action to perform.
</objective>

<available_actions>
Click [Numerical_Label]           # Click on a specific element
Type [Numerical_Label]; [Content] # Type content into a text field
Scroll [WINDOW]; [up/down]        # Scroll the entire page
Scroll [Numerical_Label]; [up/down] # Scroll within a specific element
Wait                             # Wait for page to load
GoBack                           # Go back to previous page
Google                           # Return to Google search
ANSWER; [content]                # Provide final answer
</available_actions>

<action_guidelines>
1. Execute only one action per iteration
2. When clicking or typing, ensure to select the correct bounding box
3. Numeric labels lie in the top-left corner of their corresponding bounding boxes
4. Do not add any extra punctuation, dashes, or periods
5. Use exact numerical labels as shown in the screenshot
</action_guidelines>

<browsing_guidelines>
1. Do not interact with login, sign-in, or donation elements
2. Select elements strategically to minimize wasted actions
3. If you can't find information after 3 attempts, try a different approach
4. Avoid repeating the same action multiple times
5. Return to Google search if current path is not productive
</browsing_guidelines>

<response_format>
Your response must follow this exact format:

Thought: (briefly explain your reasoning)
Action: (one of the actions above, exactly as shown)
</response_format>

<examples>
Example 1 - Searching:
Thought: I need to search for the weather information
Action: Type 7 weather in New York

Example 2 - Clicking Result:
Thought: I can see the search result I want
Action: Click 3

Example 3 - Navigation:
Thought: I need to return to the search page
Action: GoBack

Example 4 - Final Answer:
Thought: I have found the requested information
Action: ANSWER; The temperature in New York is 72°F
</examples>

<error_handling>
1. If a page is loading, use Wait action
2. If stuck or lost, use Google action to restart
3. If content not visible, try Scroll action
4. If information found but incomplete, continue searching
5. If task seems impossible after multiple attempts, provide ANSWER explaining why
</error_handling>
"""

    bbox_descriptions = format_descriptions(state["bboxes"])
    
    try:
        if MODEL_TYPE == "ollama":
            # Format for Ollama
            messages = [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": f"{state['input']}\n\nHere are the elements on the page:\n{bbox_descriptions}",
                    "images": [state['img']]  # Ollama expects base64 images directly in an images array
                }
            ]
            
            try:
                print("Sending request to Ollama...")
                response = await client.chat(
                    model=VISION_MODEL,
                    messages=messages,
                    options={'temperature': 0}
                )
                print(f"Ollama response received: {response.message.content}")
                
                # Validate response format
                content = response.message.content
                if not content or not isinstance(content, str):
                    raise ValueError(f"Invalid response format from Ollama: {content}")
                
                # Check if response contains required format
                if "Action:" not in content:
                    raise ValueError(f"Response missing Action format: {content}")
                
                return parse_action(content)
            except Exception as ollama_error:
                print(f"Ollama error: {ollama_error}")
                # Try to recover with a retry
                return {"action": "retry", "args": f"Ollama error: {str(ollama_error)}"}
        else:
            # Format for OpenAI
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {
                        "type": "text",
                        "text": state["input"]
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{state['img']}"
                        }
                    },
                    {
                        "type": "text", 
                        "text": bbox_descriptions
                    }
                ]}
            ]
            
            response = client.chat.completions.create(
                model=VISION_MODEL,
                messages=messages,
                max_tokens=1000
            )
            return parse_action(response.choices[0].message.content)
    except Exception as e:
        print(f"Error in get_agent_action: {e}")
        return {"action": "retry", "args": f"Error in LLM call: {str(e)}"}

################################################
# Agent State Management
################################################
def select_tool(state: AgentState):
    action = state['prediction']['action']
    
    # Initialize counters
    if 'action_count' not in state:
        state['action_count'] = {}
    if 'visited_urls' not in state:
        state['visited_urls'] = []
    
    # Track URL
    current_url = state['page'].url
    if current_url not in state['visited_urls']:
        state['visited_urls'].append(current_url)
    
    # Count actions
    state['action_count'][action] = state['action_count'].get(action, 0) + 1
    
    # Check limits
    if state['action_count'].get(action, 0) > 5:
        return None
    if len(state['visited_urls']) > 10:
        return None
    if state['step'] > 30:
        return None
    
    if action == 'ANSWER':
        return None
    if action == 'retry':
        return 'agent'
        
    return action

tools = {
    "Click": click,
    "Type": type_text,
    "Scroll": scroll,
    "Wait": wait,
    "GoBack": go_back,
    "Google": to_google,
}

################################################
# Main Execution
################################################
async def setup_sandbox_browser():
    browser = await async_playwright().start()
    browser = await browser.chromium.launch(headless=False)
    page = await browser.new_page()
    await page.goto("https://www.google.com")
    return page

async def call_agent(question: str, page: Page, client: OpenAI, max_steps: int = 30):
    state = {
        "page": page,
        "input": question,
        "step": 0,
        "action_count": {},
        "visited_urls": [],
        "observation": "",
        "bboxes": [],
        "img": "",
        "prediction": None
    }
    
    steps = []
    while state["step"] < max_steps:
        try:
            # Wait for page to load
            await page.wait_for_load_state("networkidle")
            
            # Get page state
            marked_page = await mark_page(page)
            state.update(marked_page)
            
            # Get agent action
            prediction = await get_agent_action(client, state)
            state["prediction"] = prediction
            
            action = prediction["action"].rstrip(';')
            step = f"{len(steps) + 1}. {action}: {prediction['args']}"
            steps.append(step)
            print(step)
            
            # Save agent data
            agent_data_path = "./Tools/WebVoyager/AgentNotebook/agent_data.txt"
            os.makedirs(os.path.dirname(agent_data_path), exist_ok=True)
            with open(agent_data_path, "a+") as file:
                state_copy = dict(state)
                if 'img' in state_copy:
                    state_copy['img'] = '...'
                file.write(f"{json.dumps(state_copy, default=str)}\n")

            # Handle action
            if action == "ANSWER":
                break
                
            tool = tools.get(action)
            if tool:
                try:
                    # Wait briefly before action
                    await asyncio.sleep(1)
                    observation = await tool(state)
                    state["observation"] = observation
                    
                    # Wait for any navigation to complete
                    await page.wait_for_load_state("networkidle")
                except playwright.async_api.Error as e:
                    print(f"Playwright error: {e}")
                    if "context was destroyed" in str(e):
                        # Wait and retry once
                        await asyncio.sleep(2)
                        try:
                            observation = await tool(state)
                            state["observation"] = observation
                        except Exception as retry_e:
                            print(f"Retry failed: {retry_e}")
                            break
            else:
                print("Action not recognized.")
                break
                
            state["step"] += 1
            
            # Wait briefly between actions
            await asyncio.sleep(1)
            
        except Exception as e:
            print(f"Error: {e}")
            break
    
    # Save steps
    steps_path = "./Tools/WebVoyager/AgentNotebook/agent_steps.txt"
    os.makedirs(os.path.dirname(steps_path), exist_ok=True)
    with open(steps_path, "w") as file:
        for step in steps:
            file.write(f"{step}\n")
            
    return steps

async def execute_query(query: str):
    page = None
    try:
        page = await setup_sandbox_browser()
        res = await call_agent(query, page, client)
        print(f"Final response: {res[-1]}")
        return res[-1]
    finally:
        if page:
            await page.close()
            print("Browser closed.")
