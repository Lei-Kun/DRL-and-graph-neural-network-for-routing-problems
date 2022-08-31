

class Depot:

    def __init__(self, max_vehicles, max_duration, max_load):
        self.pos = (0, 0)
        self.max_vehicles = max_vehicles
        self.max_duration = max_duration
        self.max_load = max_load

        self.closest_customers = []


class Customer:

    def __init__(self, cid, x, y, service_duration, demand):
        self.id = cid
        self.pos = (x, y)
        self.service_duration = service_duration
        self.demand = demand
