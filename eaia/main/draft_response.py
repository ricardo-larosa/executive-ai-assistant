"""Core agent responsible for drafting email."""

from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.store.base import BaseStore

from eaia.schemas import (
    State,
    NewEmailDraft,
    ResponseEmailDraft,
    Question,
    Ignore,
    email_template,
)
from eaia.main.config import get_config

EMAIL_WRITING_INSTRUCTIONS = """You are {full_name}'s email assistant. You are a top-notch email assistant who cares about {name} performing as well as possible.

{background}

{name} gets lots of emails. This has been determined to be an email that is worth {name} responding to.

Your job is to help {name} respond. You can do this in a few ways.

# Using the `Question` tool

First, get all required information to respond. You can use the Question tool to ask {name} for information if you do not know it.

When drafting emails (either to response on thread or , if you do not have all the information needed to respond in the most appropriate way, call the `Question` tool until you have that information. Do not put placeholders for names or emails or information - get that directly from {name}!
You can get this information by calling `Question`. Again - do not, under any circumstances, draft an email with placeholders or you will get fired.

If people ask {name} if he can attend some event or meet with them, do not agree to do so unless he has explicitly okayed it!

Remember, if you don't have enough information to respond, you can ask {name} for more information. Use the `Question` tool for this.
Never just make things up! So if you do not know something, or don't know what {name} would prefer, don't hesitate to ask him.

# Using the `ResponseEmailDraft` tool

Next, if you have enough information to respond, you can draft an email for {name}. Use the `ResponseEmailDraft` tool for this.

## `ResponseEmailDraft` rules
- ALWAYS draft emails as if they are coming from {name}. 
- Never draft them as "{name}'s assistant" or someone else.
- When adding new recipients - only do that if {name} explicitly asks for it and you know their emails. If you don't know the right emails to add in, then ask {name}. You do NOT need to add in people who are already on the email! Do NOT make up emails.

{response_preferences}

# Using the `NewEmailDraft` tool

Sometimes you will need to start a new email thread. If you have all the necessary information for this, use the `NewEmailDraft` tool for this.

## `NewEmailDraft` rules
- If {name} asks someone if it's okay to introduce them, and they respond yes, you should draft a new email with that introduction.

"""

draft_prompt = """{instructions}

Remember to call a tool correctly! Use the specified names exactly - not add `functions::` to the start. Pass all required arguments.

Here is the email thread. Note that this is the full email thread. Pay special attention to the most recent email.

{email}"""


async def draft_response(state: State, config: RunnableConfig, store: BaseStore):
    """Write an email to a customer."""
    model = config["configurable"].get("model", "gpt-4o")
    llm = ChatOpenAI(
        model=model,
        temperature=0.5,
        parallel_tool_calls=False,
        tool_choice="required",
    )
    tools = [
        NewEmailDraft,
        ResponseEmailDraft,
        Question,
    ]
    messages = state.get("messages") or []
    if len(messages) > 0:
        tools.append(Ignore)
    prompt_config = get_config(config)
    namespace = (config["configurable"].get("assistant_id", "default"),)
    key = "response_preferences"
    result = await store.aget(namespace, key)
    if result and "data" in result.value:
        response_preferences = result.value["data"]
    else:
        await store.aput(namespace, key, {"data": prompt_config["response_preferences"]})
        response_preferences = prompt_config["response_preferences"]
    _prompt = EMAIL_WRITING_INSTRUCTIONS.format(
        response_preferences=response_preferences,
        name=prompt_config["name"],
        full_name=prompt_config["full_name"],
        background=prompt_config["background"],

    )
    input_message = draft_prompt.format(
        instructions=_prompt,
        email=email_template.format(
            email_thread=state["email"]["page_content"],
            author=state["email"]["from_email"],
            subject=state["email"]["subject"],
            to=state["email"].get("to_email", ""),
        ),
    )

    model = llm.bind_tools(tools)
    messages = [{"role": "user", "content": input_message}] + messages
    i = 0
    while i < 5:
        response = await model.ainvoke(messages)
        if len(response.tool_calls) != 1:
            i += 1
            messages += [{"role": "user", "content": "Please call a valid tool call."}]
        else:
            break
    return {"draft": response, "messages": [response]}
