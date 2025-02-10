#!/usr/bin/env python3


class LogMsg:
    
    LEVEL = "DEBUG"

    def DEBUG(func):
        def wrapper(*args, **kwargs):
            if LogMsg.LEVEL in ["DEBUG"]:
                print(f"\033[0m[DEBUG]: {func(*args, **kwargs)}\033[0m")
        return wrapper
     
    def ALERT(func):
        def wrapper(*args, **kwargs):
            if LogMsg.LEVEL in ["DEBUG", "ALERT"]:
                print(f"\033[38;2;255;165;0m[ALERT]: {func(*args, **kwargs)}\033[0m")
        return wrapper
    
    def FATAL(func):
        def wrapper(*args, **kwargs):
            if LogMsg.LEVEL in ["DEBUG", "ALERT", "FATAL"]:
                print(f"\033[38;2;255;0;0m[FATAL]: {func(*args, **kwargs)}\033[0m")
        return wrapper  

    # messages -----------------------------------------------------------------
    @ DEBUG
    def message1(var1): return (
        f"some message debug {var1=}"
    )
    
    @ ALERT
    def message2(): return (
        f"some message alert"
    )
    
    @FATAL
    def message3(): return (
        f"some message fatal"
    )
    
LogMsg.LEVEL = "DEBUG"
LogMsg.message1(1243)
LogMsg.message2()
LogMsg.message3()

LogMsg.LEVEL = "ALERT"
LogMsg.message1(124)
LogMsg.message2()
LogMsg.message3()

LogMsg.LEVEL = "FATAL"
LogMsg.message1(142)
LogMsg.message2()
LogMsg.message3()
