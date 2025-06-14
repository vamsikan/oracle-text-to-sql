num=10
def func1():
    #global num
    num=12
    def func2():       
        nonlocal num
        num=7
        print(num)
    func2()
    print(num)
func1()
print(num)

x=4

def fun():

    a=4

    global x

    x=4

    

    print(a)

fun()


def fun():

    x=x+1

    print(x)

 

   

x=12

print(x)