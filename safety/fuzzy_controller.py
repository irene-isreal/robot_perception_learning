class FuzzySafetyController:
    """
    Fuzzy-logic obstacle avoidance controller.

    Actions:
        0 = Forward
        1 = Backward
        2 = Turn left
        3 = Turn right
    """

    def __init__(self, max_range=180):
        self.max_range = max_range
        self.last_turn = 3

    def clamp(self, value):
        return max(0.0, min(1.0, value))

    def memberships(self, distance):
        distance = max(0, min(self.max_range, distance))

        if distance <= 25:
            near = 1.0
        elif distance < 55:
            near = (55 - distance) / 30
        else:
            near = 0.0

        if 25 < distance <= 85:
            medium = (distance - 25) / 60
        elif 85 < distance < 145:
            medium = (145 - distance) / 60
        else:
            medium = 0.0

        if distance <= 70:
            far = 0.0
        elif distance < 140:
            far = (distance - 70) / 70
        else:
            far = 1.0

        return {
            "near": self.clamp(near),
            "medium": self.clamp(medium),
            "far": self.clamp(far)
        }

    def choose_turn(self, left_distance, right_distance):
        if left_distance > right_distance:
            action = 2

        elif right_distance > left_distance:
            action = 3

        else:
            action = 2 if self.last_turn == 3 else 3

        self.last_turn = action
        return action

    def get_action(
        self,
        front_distance,
        left_distance,
        right_distance
    ):
        front = self.memberships(front_distance)
        left = self.memberships(left_distance)
        right = self.memberships(right_distance)

        forward_score = front["far"]

        forward_score = max(
            forward_score,
            min(
                front["medium"],
                left["far"],
                right["far"]
            ) * 0.8
        )

        left_score = min(
            front["near"],
            left["far"]
        )

        left_score = max(
            left_score,
            min(
                front["medium"],
                left["far"],
                right["near"]
            )
        )

        right_score = min(
            front["near"],
            right["far"]
        )

        right_score = max(
            right_score,
            min(
                front["medium"],
                right["far"],
                left["near"]
            )
        )

        if front["near"] > 0:
            left_score = max(
                left_score,
                min(front["near"], left["medium"]) * 0.5
            )

            right_score = max(
                right_score,
                min(front["near"], right["medium"]) * 0.5
            )

        scores = {
            0: forward_score,
            2: left_score,
            3: right_score
        }

        # If no fuzzy rule supports any action,
        # do not automatically choose forward.
        if max(scores.values()) <= 0:

            if front_distance < 85:
                action = self.choose_turn(
                    left_distance,
                    right_distance
                )
            else:
                action = 0

            return {
                "action": action,
                "scores": scores,
                "front_memberships": front,
                "left_memberships": left,
                "right_memberships": right
            }

        maximum_score = max(scores.values())

        tied_actions = [
            action
            for action, score in scores.items()
            if abs(score - maximum_score) < 1e-9
        ]

        if len(tied_actions) > 1:

            if 0 in tied_actions:
                action = 0

            else:
                action = self.choose_turn(
                    left_distance,
                    right_distance
                )

        else:
            action = max(scores, key=scores.get)

        if action in (2, 3):
            self.last_turn = action

        return {
            "action": action,
            "scores": scores,
            "front_memberships": front,
            "left_memberships": left,
            "right_memberships": right
        }