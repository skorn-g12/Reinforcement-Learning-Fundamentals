class Stack:
    def __init__(self, arr):
        self.a = arr

    # Stack operations
    def pop(self):  # Return the top most element
        # elem = self.a[len(self.a) - 1]
        del self.a[len(self.a) - 1]
        # return elem

    def push(self, elem):  # Push on top of stack
        self.a.append(elem)

    def peek(self):  # just gimme the top most element
        return self.a[len(self.a) - 1]


if __name__ == "__main__":
    stack = Stack([])
    arr = [3, 1, 2, 5, 4]
    previousLess = [-1] * len(arr)
    for idx, elem in enumerate(arr):
        print("elem ", elem)
        while len(stack.a) > 0:
            print("while true")
            if arr[stack.peek()] > elem:
                print("popping")
                stack.pop()
            else:
                break
        print("Before pushing", stack.a)
        print("push ", idx)
        stack.push(idx)
        print("after pushing", stack.a)
        if len(stack.a) > 1:
            previousLess[idx] = stack.a[len(stack.a) - 2]

    results = [0] * len(arr)
    for idx, elem in enumerate(arr):
        prevResult = 0
        if previousLess[idx] == -1:
            prevResult = 0
        else:
            prevResult = results[previousLess[idx]]
        results[idx] = prevResult + (idx - previousLess[idx]) * elem

    print(sum(results))
