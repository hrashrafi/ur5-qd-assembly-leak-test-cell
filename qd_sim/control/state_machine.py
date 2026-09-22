"""
Minimal finite-state machine for the multi-step task sequences (pick,
transport, leak test). Deliberately small: a sequence is just a dict of
{state_name: handler}, where each handler inspects the current sim state and
returns the next state name - the same name to keep waiting, or "DONE" to
finish.
"""

from dataclasses import dataclass, field


@dataclass
class StateMachine:
    handlers: dict          # {state_name: callable(ctx) -> next_state_name}
    start_state: str
    state: str = field(init=False)

    def __post_init__(self):
        self.reset()

    def reset(self):
        self.state = self.start_state

    def tick(self, ctx):
        """Run the current state's handler once and move to the state it
        returns. A finished machine stays finished."""
        if not self.is_done():
            self.state = self.handlers[self.state](ctx)
        return self.state

    def is_done(self):
        return self.state == "DONE"
