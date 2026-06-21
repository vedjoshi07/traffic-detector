class DensityClassifier:
    def __init__(self, thresholds: dict, hysteresis_frames: int):
        self.thresholds = thresholds
        self.hysteresis_frames = hysteresis_frames
        
        # State: zone_id -> {"current_level": str, "candidate_level": str, "candidate_streak": int}
        self.states = {}

    def _get_raw_level(self, count: int) -> str:
        if count >= self.thresholds.get("high", 10):
            return "HIGH"
        elif count >= self.thresholds.get("medium", 5):
            return "MEDIUM"
        else:
            return "LOW"

    def classify(self, zone_id: str, count: int) -> str:
        raw_level = self._get_raw_level(count)
        
        if zone_id not in self.states:
            self.states[zone_id] = {
                "current_level": raw_level,
                "candidate_level": raw_level,
                "candidate_streak": 0
            }
            return raw_level
            
        state = self.states[zone_id]
        
        if raw_level == state["current_level"]:
            state["candidate_level"] = raw_level
            state["candidate_streak"] = 0
        else:
            if raw_level == state["candidate_level"]:
                state["candidate_streak"] += 1
                if state["candidate_streak"] >= self.hysteresis_frames:
                    state["current_level"] = raw_level
                    state["candidate_streak"] = 0
            else:
                state["candidate_level"] = raw_level
                state["candidate_streak"] = 1
                
        return state["current_level"]
