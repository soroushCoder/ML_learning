def main():
    print("=" * 50)
    print("ML Learning Project")
    print("=" * 50)
    print("\nAvailable modules:")
    print("1. Prediction (RNN - Gold Price Prediction)")
    print("2. ML Perceptron (Curved Decision Boundary)")
    print("3. Exit")

    while True:
        choice = input("\nEnter your choice (1-3): ").strip()

        if choice == "1":
            print("\nRunning Prediction module...")
            try:
                import predictaion
                # The predictaion.py file runs automatically when imported
            except ImportError as e:
                print(f"Error importing predictaion module: {e}")
            except Exception as e:
                print(f"Error running predictaion module: {e}")
            break

        elif choice == "2":
            print("\nRunning ML Perceptron module...")
            try:
                import ML_Percepetron
                # The ML_Percepetron.py file runs automatically when imported
            except ImportError as e:
                print(f"Error importing ML_Percepetron module: {e}")
            except Exception as e:
                print(f"Error running ML_Percepetron module: {e}")
            break

        elif choice == "3":
            print("\nExiting... Goodbye!")
            break

        else:
            print("Invalid choice. Please enter 1, 2, or 3.")


if __name__ == "__main__":
    main()
