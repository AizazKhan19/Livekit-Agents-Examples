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

class AudioPlayerAgent(Agent):
    def __init__(self)->None:
        super().__init__(
            instructions="""You are a helpful assistant communicating through voice. Don't use any unpronouncable characters.
                If asked to play audio, use the `play_audio_file` function."""
        )

    @function_tool
    async def play_audio_file(self, context: RunContext):
            """Play a local audio file"""
            audio_file_path = Path(__file__).parent / "audio.wav"

            with wave.open(str(audio_file_path), "rb") as wav_file:
                # livekit needs 3 specifications to play and audio file. channel, sample rate, frames
                num_channels = wav_file.getnchannels() # will return number of channel like 1 or 2 
                sample_rate = wav_file.getframerate() # will return audio quality frequency (16000Hz or 44100Hz)
                frames = wav_file.readframes(wav_file.getnframes())  # actual raw data (bytes) or audio file


            # now preparing audio fram for livekit
            audio_frame = rtc.AudioFrame(
            data=frames,
            sample_rate=sample_rate,
            num_channels=num_channels,
            samples_per_channel=wav_file.getnframes()

            )
            
            async def audio_generator():
                yield audio_frame   # provides the data to livekit in kind of chunks not all at once via webrtc

            await self.session.say("I am playing audio file", audio = audio_generator())
            return None, "I've played the audio file for you."
        
    async def on_enter(self):
        self.session.generate_reply()


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

    await session.start(agent=AudioPlayerAgent(), room=ctx.room)
    await ctx.connect()

if __name__ == "__main__":
    cli.run_app(server)