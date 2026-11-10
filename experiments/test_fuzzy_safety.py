
from safety.fuzzy_controller import FuzzySafetyController


def main():

    controller = FuzzySafetyController(max_range=180)

    test_cases = [
        {
            "name": "Clear path",
            "front": 170,
            "left": 150,
            "right": 150,
            "expected": 0
        },
        {
            "name": "Obstacle ahead, left clear",
            "front": 20,
            "left": 160,
            "right": 20,
            "expected": 2
        },
        {
            "name": "Obstacle ahead, right clear",
            "front": 20,
            "left": 20,
            "right": 160,
            "expected": 3
        },
        {
            "name": "Moderate front distance",
            "front": 110,
            "left": 150,
            "right": 150,
            "expected": 0
        },
        {
            "name": "Obstacle ahead, both sides open",
            "front": 15,
            "left": 150,
            "right": 120,
            "expected": 2
        },
        {
            "name": "Obstacle ahead, both sides blocked",
            "front": 15,
            "left": 15,
            "right": 15,
            "expected": None
        }
    ]

    action_names = {
        0: "FORWARD",
        1: "BACKWARD",
        2: "LEFT",
        3: "RIGHT"
    }

    passed = 0

    print("\n========== FUZZY SAFETY TESTS ==========")

    for case in test_cases:

        result = controller.get_action(
            case["front"],
            case["left"],
            case["right"]
        )

        action = result["action"]

        print("\nTest:", case["name"])

        print(
            "Distances | Front:",
            case["front"],
            "Left:",
            case["left"],
            "Right:",
            case["right"]
        )

        print(
            "Fuzzy scores:",
            {
                action_names[k]: round(v, 3)
                for k, v in result["scores"].items()
            }
        )

        if case["expected"] is None:
            # When every direction is blocked, the controller
            # should choose a turning direction, not move forward.
            success = action in (2, 3)
        else:
            success = action == case["expected"]

        print("Selected action:", action_names[action])

        if success:
            print("Result: PASS")
            passed += 1
        else:
            print("Result: FAIL")

    print("\n========== SUMMARY ==========")
    print("Passed:", passed, "/", len(test_cases))

    if passed == len(test_cases):
        print("All fuzzy controller tests passed!")
    else:
        print("Some tests failed. Review the controller.")


if __name__ == "__main__":
    main()
