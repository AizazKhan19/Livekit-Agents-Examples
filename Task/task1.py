from livekit.agents import Agent, AgentTask, AgentServer, AgentSession, cli, inference, function_tool, get_job_context, JobProcess, JobContext
from dotenv import load_dotenv
from livekit.plugins import silero
import logging


logger = load_dotenv()

logger = logging.getLogger("Simple-Task")
logger.setLevel(logging.INFO)

#defining the agent's task
class NameCollectorTask(AgentTask[str]):
    def __init__(self):

        # this is the task instruction
        super().__init__(
            instructions= """Collect only the user's name. Nothing more nor less.""",
        )

    async def on_enter(self):
        # when executing this task, this is what agent will talk like
        await self.session.generate_reply(
            instructions= " Ask poilitely the name only form the user",
        )

    @function_tool
    async def name_collected(self, name:str)->None:
        """ Execute when the user provided you the name"""
        self.complete(name)

    @function_tool
    async def name_not_given(self)->None:
        """Execute when user denied of giving you the name"""
        self.complete("")

# defining the agent

class AskNameAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions= " you are a helpful assistant that only collects the user name and provide information only when user give name else you do not provide information.",

        )

    async def on_enter(self):
        name = await NameCollectorTask()
        if name:
            await self.session.generate_reply(instructions= " provide user on the topic ")
        else:
            await self.session.generate_reply(instructions=" Do not provide information to the user on topic")
            job_ctx = get_job_context()
            await job_ctx.delete_room()


#server initialization
server = AgentServer()

def prewarm(proc: JobProcess ):
    proc.userdata['vad']=silero.VAD.load()
server.setup_fnc = prewarm

@server.rtc_session()
async def enterypoint(ctx: JobContext):
    ctx.log_context_fields = {"room":ctx.room.name}


    #starting the session
    session = AgentSession(
        llm = inference.LLM("openai/gpt-4o-mini"),
        stt = inference.STT("deepgram/nova-3"),
        tts = inference.TTS("cartesia/sonic-3"),
        vad = ctx.proc.userdata['vad'],
        preemptive_generation=True,
    )

    session.start(agent=AskNameAgent(), room= ctx.room)
    await ctx.connect()
    
if __name__ == "__main__":
    cli.run_app(server)