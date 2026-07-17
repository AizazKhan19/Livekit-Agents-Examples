import logging
from pathlib import Path
import wave
from dotenv import load_dotenv
from livekit.agents import JobContext, JobProcess, AgentServer,Agent, cli, AgentSession, inference, RunContext, function_tool
from livekit.plugins import silero
# from livekit.plugins import groq
from livekit import rtc


load_dotenv()

logger = logging.getLogger("playing-audio")
logger.setLevel(logging.INFO)

class RepeaterAgent(Agent):
    def __init__(self)->None:
        super().__init__(
            instructions="""You are a helpful assistant that repeats what the user says"""
        )


server = AgentServer()

def prewarm(proc: JobProcess ):
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

    #event listener decorator 
    @session.on("user_input_transcribed")
    def on_transcript(transcript):
        if transcript.is_final:
            session.say(transcript.transcript)
            

    await session.start(agent=RepeaterAgent(), room=ctx.room)
    await ctx.connect()

if __name__ == "__main__":
    cli.run_app(server)