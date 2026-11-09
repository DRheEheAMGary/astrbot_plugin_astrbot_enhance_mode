from collections import OrderedDict, defaultdict

from .batch_scheduler import BatchCoordinator


class RuntimeState:
    def __init__(self) -> None:
        self.session_chats: dict[str, list[str]] = defaultdict(list)
        self.active_reply_stacks: dict[str, list[str]] = defaultdict(list)
        self.model_choice_histories: dict[str, list[str]] = defaultdict(list)
        self.image_message_registry: dict[str, dict[str, dict[str, object]]] = (
            defaultdict(dict)
        )
        self.batch_coordinators: dict[str, BatchCoordinator] = {}
        self.origin_lru: OrderedDict[str, None] = OrderedDict()

    def _evict_origin_state(self, origin: str) -> None:
        self.session_chats.pop(origin, None)
        self.active_reply_stacks.pop(origin, None)
        self.model_choice_histories.pop(origin, None)
        self.image_message_registry.pop(origin, None)
        coordinator = self.batch_coordinators.pop(origin, None)
        if coordinator is not None:
            coordinator.close()

    def touch_origin(self, origin: str, max_origins: int) -> None:
        if not origin:
            return
        self.origin_lru.pop(origin, None)
        self.origin_lru[origin] = None
        while len(self.origin_lru) > max_origins:
            oldest, _ = self.origin_lru.popitem(last=False)
            self._evict_origin_state(oldest)

    def cleanup_origin(self, origin: str) -> None:
        self._evict_origin_state(origin)
        self.origin_lru.pop(origin, None)

    def get_batch_coordinator(self, origin: str) -> BatchCoordinator:
        coordinator = self.batch_coordinators.get(origin)
        if coordinator is None:
            coordinator = BatchCoordinator(origin)
            self.batch_coordinators[origin] = coordinator
        return coordinator

    def close_all_batch_coordinators(self) -> None:
        for coordinator in self.batch_coordinators.values():
            coordinator.close()
        self.batch_coordinators.clear()
