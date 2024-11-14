# Prompt the user to type "yes" to continue
response = input("Type 'yes' to continue: ")

# Keep prompting until the user types "yes"
while response.lower() != "yes":
    response = input("Please type 'yes' to continue: ")

print("Continuing with the program...")
# Your code here
