import streamlit as st

class AlertManager:
    def __init__(self, alert_level: str):
        self.alert_level = alert_level
        self.active_alerts = {}  # zone_id -> bool
        self.previous_levels = {} # zone_id -> str
        self.just_triggered = [] # list of zone_ids that triggered an alert THIS frame

    def check(self, zone_id: str, new_level: str):
        prev_level = self.previous_levels.get(zone_id, "LOW")
        
        # Edge triggered: went INTO the alert level
        if new_level == self.alert_level and prev_level != self.alert_level:
            self.active_alerts[zone_id] = True
            self.just_triggered.append(zone_id)
        
        # Turn off alert if level drops
        if new_level != self.alert_level:
            self.active_alerts[zone_id] = False
            
        self.previous_levels[zone_id] = new_level

    def get_active_alerts(self) -> list:
        return [zone_id for zone_id, is_active in self.active_alerts.items() if is_active]

    def get_and_clear_just_triggered(self) -> list:
        triggered = self.just_triggered.copy()
        self.just_triggered.clear()
        return triggered
