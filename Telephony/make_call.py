import logging
from dotenv import load_dotenv
from livekit.agents import JobContext, JobProcess, Agent, AgentSession, AgentServer, cli, inference
from livekit.plugins import silero

load_dotenv()

logger = logging.getLogger("answer-call")
logger.setLevel(logging.INFO)



server = AgentServer()
def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

server.setup_fnc = prewarm


class AnswerCallAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions = """
                            You are a helpful assistant that answers the incoming call and provide information to the user on the topic."""
        )


    async def on_enter(self):
        await self.session.generate_reply()


@server.rtc_session(agent_name="inbound_call_agent")
async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}


    session = AgentSession(
        stt=inference.STT(model="deepgram/nova-3-general"),
        llm=inference.LLM(model="openai/gpt-4.1-mini"),
        tts=inference.TTS(model="cartesia/sonic-3", voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    agent = AnswerCallAgent()
    session.start(agent=agent, room=ctx.room)
    await ctx.connect()

if __name__ == "__main__":
    cli.run_app(server)