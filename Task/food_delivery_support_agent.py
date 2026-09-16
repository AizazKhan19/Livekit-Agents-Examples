from livekit.agents import Agent, AgentServer, AgentSession, JobContext, function_tool, get_job_context, AgentTask, JobProcess, inference, cli
from livekit.plugins import silero
from dotenv import load_dotenv

load_dotenv()

class FoodDeliverySupportTask(AgentTask[bool]):
    def __init__(self, chat_ctx = None):
        super().__init__(
            instructions = """ Ask for order id and phone number and get a order id and phone number. Be polite and professional.""",
            chat_ctx = chat_ctx
        )

    async def on_enter(self)-> None:
        await self.session.generate_reply(
            instructions = """ Briefly introduce yourself, then ask for order id and phone number. Make it clear that they can decline."""
        )

    @function_tool()
    async def order_info_provided(self, order_id: str, phone_number: str)->None:
        """Use this when the user provided order id and phone number."""
        print(f"Order ID: {order_id}, Phone Number: {phone_number}")
        self.complete(True)

    @function_tool()
    async def order_info_not_provided(self)->None:
        """Use this when the user denies to provide order id and phone number."""
        self.complete(False)

class FoodDeliverySupportAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="You are a friendly food delivery support representative.",

        )

    async def on_enter(self)-> None:
            result = await FoodDeliverySupportTask(chat_ctx=self.chat_ctx.copy(exclude_instructions=True))
            if result:
                await self.session.generate_reply(instructions="Offer your assistance to the user.")
            else:
                await self.session.generate_reply(instructions="Inform the user that you are unable to proceed and will end the call.")
                job_ctx = get_job_context()
                await job_ctx.delete_room()


server = AgentServer()

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

server.setup_fnc = prewarm

@server.rtc_session(agent_name="food_delivery_support_agent")
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
    await session.start(agent=FoodDeliverySupportAgent(), room=ctx.room)  

if __name__ == "__main__":
    cli.run_app(server)