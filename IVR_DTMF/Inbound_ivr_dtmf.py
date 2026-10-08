import asyncio
from livekit import rtc
from livekit.agents import Agent, AgentServer, AgentSession, JobContext, JobProcess, inference, cli
from livekit.agents.beta.workflows.dtmf_inputs import GetDtmfTask

from livekit.plugins import silero, elevenlabs
import dotenv

dotenv.load_dotenv('.env')

class BookingAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="""
            you are a friendly booking assitant that helps users book appointments using IVR.
            """
        )

    async def on_enter(self):
        await self.session.generate_reply(instructions= 
            """ Welcome to our company.
            Press 1 for Booking Service.
            Press 2 for Information Desk."""
            )
        await asyncio.sleep(2)

        

        
server = AgentServer()

def prewarm(proc : JobProcess):
    proc.userdata["vad"] = silero.VAD.load()
server.setup_fnc = prewarm

@server.rtc_session(agent_name="booking_agent")
async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}

    



    session = AgentSession(
        stt = inference.STT(model = "deepgram/nova-3-general"),
        tts = elevenlabs.TTS(model = "eleven_v3", voice_id= "IKne3meq5aSn9XLyUdCD"),
        llm = inference.LLM(model = "openai/gpt-4o-mini"),
        vad = ctx.proc.userdata['vad'],
        preemptive_generation=True,
        ivr_detection=False
    )

    await ctx.connect()
    @ctx.room.on("sip_dtmf_received")
    def dtmf_received(dtmf: rtc.SipDTMF):

        print("DTMF:", dtmf.digit)

        if dtmf.digit == "1":
            session.generate_reply(
                instructions="Welcome to Booking Service."
            )

        elif dtmf.digit == "2":
            session.generate_reply(
                instructions="Welcome to Information Desk."
            )
    await session.start(agent= BookingAgent(), room=ctx.room)  

if __name__ == "__main__":
    cli.run_app(server)