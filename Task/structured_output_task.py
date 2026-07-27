from livekit.agents import cli, inference, JobContext, JobProcess, Agent, AgentTask, get_job_context, AgentServer, AgentSession
from dotenv import load_dotenv
from livekit.plugins import silero
import logging
from dataclasses import dataclass

load_dotenv()

logger = logging.getLogger("playing-audio")
logger.setLevel(logging.INFO)


# making custom output stucture
@dataclass
class CustomerData :
    name : str
    email : str
    address : str



# now defining my task
class InfoCollectionTask(AgentTask[CustomerData]):
    def __init__(self, chat_ctx=None):
        super().__init__(
            instructions= " Ask for name, email and address. Be polite and professional. ",
            chat_ctx=chat_ctx,
        )

    async def on_enter(self):
        await self.session.generate_reply(
            instructions= " Your Name is Rose, Briefly introduce yourself, then gently ask for user's data. "
        )

# now defining my agent
class InfoDeskAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions= " You are a helpful information desk agent. Your job is to ask for user's name, email and address before giving them any information. ",

        )

    async def on_enter(self)->None:
            result = await InfoCollectionTask(chat_ctx= self.chat_ctx)
            if result:
                await self.session.generate_reply( instructions = f"offer your assistance to the {result.name}")

            else:
                await self.session.generate_reply(instructions = " Inform the user that you can not proceed and will end the call ")
                job_ctx = get_job_context()
                await job_ctx.delete_room()


server = AgentServer()

def prewarm(proc: JobProcess):
    proc.userdata['vad'] = silero.VAD.load()

server.setup_fnc = prewarm

@server.rtc_session()
async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room":ctx.room.name}

    #initializing the session
    session = AgentSession(
            stt = inference.STT(model = "deepgram/nova-3-general"),
            tts = inference.TTS(model = "cartesia/sonic-3", voice= "9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"),
            llm = inference.LLM(model = "openai/gpt-4o-mini"),
            vad = ctx.proc.userdata['vad'],
            preemptive_generation=True,
            )
    
    await session.start(agent=InfoDeskAgent(), room=ctx.room)
    await ctx.connect()
if __name__ == "__main__":
    cli.run_app(server)
