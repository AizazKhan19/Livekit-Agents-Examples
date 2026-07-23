from livekit.agents import AgentTask, function_tool
from livekit.agents import Agent, function_tool, get_job_context, cli, inference, AgentServer, AgentSession, JobProcess, JobContext
from livekit.plugins import silero
from dotenv import load_dotenv
import logging


load_dotenv()

logger = logging.getLogger("playing-audio")
logger.setLevel(logging.INFO)



class CollectConsent(AgentTask[bool]):   # here the [bool] means the final output of the task will be boolean  True | False 
    def __init__(self, chat_ctx=None):
        super().__init__(

            # below instruction are the task instructions 
            instructions="""
            Ask for recording consent and get a clear yes or no answer.
            Be polite and professional.
            """,
            chat_ctx=chat_ctx,
        )

    async def on_enter(self) -> None:
        await self.session.generate_reply(
            # below instruction is the agents instructions ( just like role and goal )
            instructions="""
            Briefly introduce yourself, then ask for permission to record the call for quality assurance and training purposes.
            Make it clear that they can decline.
            """
        )

    @function_tool
    async def consent_given(self) -> None:
        """Use this when the user gives consent to record."""
        self.complete(True)

    @function_tool
    async def consent_denied(self) -> None:
        """Use this when the user denies consent to record."""
        self.complete(False)


# defining the agent
class CustomerServiceAgent(Agent):
    def __init__(self):
        super().__init__(instructions="You are a friendly customer service representative.")

    async def on_enter(self) -> None:
        if await CollectConsent(chat_ctx=self.chat_ctx):
            await self.session.generate_reply(instructions="Offer your assistance to the user.")
        else:
            await self.session.generate_reply(instructions="Inform the user that you are unable to proceed and will end the call.")
            job_ctx = get_job_context()
            await job_ctx.delete_room()

server = AgentServer()

def prewarm(proc: JobProcess):
    proc.userdata['vad']=silero.VAD.load()

server.setup_fnc=prewarm


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    ctx.log_context_fields={"room":ctx.room.name}
    
    session = AgentSession(
        stt = inference.STT(model = "deepgram/nova-3-general"),
        tts = inference.TTS(model = "cartesia/sonic-3", voice= "9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"),
        llm = inference.LLM(model = "openai/gpt-4o-mini"),
        vad = ctx.proc.userdata['vad'],
        preemptive_generation=True,
        )

    await session.start(agent=CustomerServiceAgent(), room=ctx.room)
    await ctx.connect()
if __name__ == "__main__":
    cli.run_app(server)