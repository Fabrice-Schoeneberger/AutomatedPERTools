class MyClass:
    def funka(self):
        print("funka method called")

# Create an instance of MyClass
obj = MyClass()

# Method name as a string
method_name = "funka"

# Call the method using getattr
method = getattr(obj, method_name)

# Call the method
method()
