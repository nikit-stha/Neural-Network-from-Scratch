import matplotlib.pyplot as plt

x = []
y1 = []
y2 = []

i = 1

while True:
    try:
        a = input(f"Enter y1 for {i} (N to stop): ")

        if a.strip().lower() == "n":
            break

        b = input(f"Enter y2 for {i} (N to stop): ")

        if b.strip().lower() == "n":
            break

        y1.append(float(a))
        y2.append(float(b))
        x.append(i)

        i += 1

    except ValueError:
        print("Please enter numbers.")

plt.plot(x, y1, marker="o", label="y1")
plt.plot(x, y2, marker="o", label="y2")

plt.xlabel("Progress")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.show()

# Compile ./network | python main.py