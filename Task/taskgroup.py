import logging
from dotenv import load_dotenv
from livekit.agents import AgentServer, AgentSession, JobContext, JobProcess, Agent, AgentTask, cli, inference, function_tool
from livekit.plugins import silero
from dataclasses import dataclass
from livekit.agents.beta.workflows import TaskGroup
from typing import Optional

logger = load_dotenv()

logger = logging.getLogger("Simple-Task")
logger.setLevel(logging.INFO)

# structure data for task
@dataclass
class GetNameResult:
    first_name : str
    last_name : str

@dataclass
class GetContactResult:
    email : Optional [str] = None
    cell_number : Optional[str] = None

@dataclass
class GetExperienceResult:
    experience : str

# defining the task
class GetNameTask(AgentTask[GetNameResult]):
    def __init__(self):
        super().__init__(
            instructions = " Collect the first and last name from the user only. Nothing more nor less. ",
        )

    async def on_enter(self):
        await self.session.generate_reply(
            instructions= " Politely ask for user's first and last name only.",
        )
    @function_tool
    async def name_provided(self, f_name : str, l_name : str)->None:
        """Execute only when user provide you first and last name only. Else do not execute ."""
        result = GetNameResult(first_name=f_name, last_name=l_name)
        self.complete(result)

    @function_tool
    async def name_not_provided(self)->None:
        """Execute when user does not provide you first and last name. Else do not execute ."""
        await self.session.generate_reply(
            instructions= " Politely ask for user's first and last name",
        )        


class GetContactTask(AgentTask[GetContactResult]):
    def __init__(self):
        super().__init__(
            instructions= "Collect email and cell number from the user. If user provide only one of them then task is completed. if Provide both then it is also okay.",
        )

    async def on_enter(self):
        await self.session.generate_reply(
            instructions= " Politely ask for email and cell number from the user.",
        )

    @function_tool
    async def provided(self, email: str , cell_no : str)-> None:
        """Execute only when user provide atleast one thing either cell number or email. Also execute this when user provide both. Else do not execute.
        if user provide only one of them then mark the other one as None"""

        result = GetContactResult(email= email, cell_number=cell_no)
        if result.email or result.cell_number:
            self.complete(result)

    @function_tool
    async def not_provided(self)->None:
        """Execute when user did no provide any of them like email and cell number"""
        await self.session.generate_reply(
                instructions=" Politely ask for email and cell number from the user.",
            )

class GetExperienceTask(AgentTask[GetExperienceResult]):
    def __init__(self):
        super().__init__(
            instructions = "Collect experience from the user only. Nothing more nor less.",
        )

    async def on_enter(self):
        await self.session.generate_reply(
            instructions= " Politely ask expereince from the user.",
        )

    @function_tool
    async def provided(self, exp : str)->None:
        """Execute when user provide you the experience. Else do not execute."""

        result = GetExperienceResult(experience=exp)
        self.complete(result)

    @function_tool
    async def not_provided(self)->None:
        """Execute when user did not provide you the experience. Else do not execute."""

        await self.session.generate_reply(
            instructions= "Politely ask experience from the user."
        )


# defining the agent

class AssistantAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions= "You are a Friendly Assistant agent that helps users and provide them information on any topic only when user provide their name, contact information and experience. Else you do not provide information.",

        )

    async def on_enter(self):
        chat_ctx=self.chat_ctx
        task_group = TaskGroup(chat_ctx=chat_ctx)

        task_group.add(
            lambda : GetNameTask(),
            id = "get_name",
            description= "Collects user's first and last name"
        )

        task_group.add(
            lambda : GetContactTask(),
            id = "get_contact",
            description= " Collect user's contact like email and cell number."
        )

        task_group.add(
            lambda : GetExperienceTask(),
            id = "get_experience",
            description= " Collects experience from the user"
        )

        results = await task_group
        task_results = results.task_results

        if task_results:
            self.session.say("Thanks for providing your information.Here is the information you provided:\n")
            print("/n -----------------Data Collected from the user-------------------/n")
            print(f'User name: {task_results["get_name"].first_name} {task_results["get_name"].last_name}')
            print(f'User email: {task_results["get_contact"].email}')
            print(f'User cell number: {task_results["get_contact"].cell_number}')
            print(f'User experience: {task_results["get_experience"].experience}')

    

#server initialization
server = AgentServer()

def prewarm(proc: JobProcess ):
    proc.userdata['vad']=silero.VAD.load()
server.setup_fnc = prewarm

@server.rtc_session(agent_name="Information_Collection_Agent")
async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}


    session = AgentSession(
        llm = inference.LLM("openai/gpt-4o-mini"),
        stt = inference.STT("deepgram/nova-3"),
        tts = inference.TTS("cartesia/sonic-3"),
        vad = ctx.proc.userdata['vad'],
        preemptive_generation=True,
    )

    await session.start(agent=AssistantAgent(), room= ctx.room)
    await ctx.connect()

if __name__ == "__main__":
    cli.run_app(server)