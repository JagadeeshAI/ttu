def max_subarray_sum(nums):
    if not nums:
        raise ValueError("nums must not be empty")

    best_sum = current_sum = nums[0]
    for num in nums[1:]:
        current_sum = max(num, current_sum + num)
        best_sum = max(best_sum, current_sum)

    return best_sum


if __name__ == "__main__":
    arr = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
    print(f"Array: {arr}")
    print(f"Maximum subarray sum: {max_subarray_sum(arr)}")
