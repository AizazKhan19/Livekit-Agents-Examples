from livekit.agents import Agent, AgentServer, AgentSession, JobContext, function_tool, get_job_context, AgentTask, JobProcess, inference, cli
from livekit.plugins import silero
from dotenv import load_dotenv
from dataclasses import dataclass

load_dotenv()


# create a task that collects user name, email and contact number.
#creating data class to hold the user data
@dataclass
class UserData:
    name : str
    email : str
    phone_number : str

class DataCollectionTask(AgentTask[UserData]):
    def __init__(self, chat_ctx = None):
        super().__init__(
            instructions = """Ask for user name, email and phone number. Be polite and professional. Do not ask for any other information.
            User may decline to provide the information""",
            chat_ctx=chat_ctx
        )

    async def on_enter(self):
        await self.session.generate_reply(
            instructions= " Ask for user name, email and phone number. Be professional but friendly."
        )

    @function_tool()
    async def user_Data_provided(self, name:str, email:str, phone_number:str):
        """Use this when the user provided name, email and phone number."""

        self.complete(UserData(name=name, email=email, phone_number=phone_number))

    @function_tool()
    async def user_data_not_provided(self):
        """Use this when the user denies to provide name, email and phone number."""
        self.complete(None)


class DataCollectionAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="You are a friendly data collection agent. You provide information to user only when you have user's name, email and phone number.",
        )

    async def on_enter(self):
        result = await DataCollectionTask(chat_ctx=self.chat_ctx.copy(exclude_instructions=True))
        if result:
            await self.session.generate_reply(instructions="Thank you for providing your information. We will use it to assist you.")
            print(f"User Data Collected: Name: {result.name}, Email: {result.email}, Phone Number: {result.phone_number}")
        else:
            await self.session.generate_reply(instructions="Thank you for your time. We will not be able to assist you without your information.")
            job_ctx = get_job_context()
            await job_ctx.delete_room()
server = AgentServer()

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

server.setup_fnc = prewarm

@server.rtc_session(agent_name="data_collection_agent")
async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}

    session = AgentSession(
        stt = inference.STT(model = "deepgram/nova-3-general"),
        tts = inference.TTS(model = "cartesia/sonic-3", voice= "9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"),
        llm = inference.LLM(model = "openai/gpt-4o-mini"),
        vad = ctx.proc.userdata['vad'],
        preemptive_generation=True,
    )

    await ctx.connect()
    await session.start(agent=DataCollectionAgent(), room=ctx.room)  

if __name__ == "__main__":
    cli.run_app(server)







    
