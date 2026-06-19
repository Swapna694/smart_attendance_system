from datetime import date

class DuplicateChecker:
    """
    In-memory set of (person_id, date) pairs already marked today.
    Resets automatically when the date changes.
    """
    def __init__(self):
        self._marked = set()
        self._current_date = date.today()

    def _refresh(self):
        today = date.today()
        if today != self._current_date:
            self._marked.clear()
            self._current_date = today

    def already_marked(self, person_id):
        self._refresh()
        return person_id in self._marked

    def mark(self, person_id):
        self._refresh()
        self._marked.add(person_id)

    def reset(self):
        self._marked.clear()