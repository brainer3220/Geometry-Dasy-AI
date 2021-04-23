import pyautogui as pag

def Jump():
    """
    Jump Function
    """
    pag.moveTo(416, 275)
    pag.mouseDown()
    time.sleep(0.05)
    pag.mouseUp()


def Retry():
    """
    Retry Funtion
    """
    pag.moveTo(240, 480)
    pag.mouseDown()
    pag.mouseUp()


def Q_Value(State, Action):
    """
    Q Value Function
    """
    reword + (Discount * max(Q_next))
    # return


def BringWindow():
    """
    Bring the emulator to the front
    """
    time.sleep(0.5)
    apple = """
    tell application "BlueStacks"
    activate
    end tell
    """
    

def GetResolution():
    """
    Function to get resolution.
    Test it when you bring the emulator's resolution coordinates.
    """
    while True:
        x, y = pag.position()
        position_str = "X: " + str(x) + "Y: " + str(y)
        BringWindow()
        print(position_str)
