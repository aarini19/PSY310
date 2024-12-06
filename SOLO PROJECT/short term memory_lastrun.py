#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2024.2.1),
    on Fri Dec  6 19:11:03 2024
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
prefs.hardware['audioLatencyMode'] = '3'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER, priority)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard

# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2024.2.1'
expName = 'short term memory'  # from the Builder filename that created this script
# information about this experiment
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'session': '001',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = True
_winSize = (1024, 768)
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='/Users/shrutigupta/Desktop/lab in psychology/SOLO PROJECT/short term memory_lastrun.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # set how much information should be printed to the console / app
    if PILOTING:
        logging.console.setLevel(
            prefs.piloting['pilotConsoleLoggingLevel']
        )
    else:
        logging.console.setLevel('warning')
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log')
    if PILOTING:
        logFile.setLevel(
            prefs.piloting['pilotLoggingLevel']
        )
    else:
        logFile.setLevel(
            logging.getLevel('info')
        )
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if PILOTING:
        logging.debug('Fullscreen settings ignored as running in pilot mode.')
    
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=0,
            winType='pyglet', allowStencil=False,
            monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height', 
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    if expInfo is not None:
        # get/measure frame rate if not already in expInfo
        if win._monitorFrameRate is None:
            win._monitorFrameRate = win.getActualFrameRate(infoMsg='Attempting to measure frame rate of screen, please wait...')
        expInfo['frameRate'] = win._monitorFrameRate
    win.mouseVisible = False
    win.hideMessage()
    # show a visual indicator if we're in piloting mode
    if PILOTING and prefs.piloting['showPilotingIndicator']:
        win.showPilotingIndicator()
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    
    # Setup iohub keyboard
    ioConfig['Keyboard'] = dict(use_keymap='psychopy')
    
    # Setup iohub experiment
    ioConfig['Experiment'] = dict(filename=thisExp.dataFileName)
    
    # Start ioHub server
    ioServer = io.launchHubServer(window=win, **ioConfig)
    
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='iohub'
        )
    if deviceManager.getDevice('keyWelcome') is None:
        # initialise keyWelcome
        keyWelcome = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='keyWelcome',
        )
    if deviceManager.getDevice('keyInstructions') is None:
        # initialise keyInstructions
        keyInstructions = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='keyInstructions',
        )
    if deviceManager.getDevice('keyWaitScreen') is None:
        # initialise keyWaitScreen
        keyWaitScreen = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='keyWaitScreen',
        )
    if deviceManager.getDevice('keyY_N') is None:
        # initialise keyY_N
        keyY_N = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='keyY_N',
        )
    if deviceManager.getDevice('key_resp') is None:
        # initialise key_resp
        key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp',
        )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    playbackComponents : list, tuple
        List of any components with a `pause` method which need to be paused.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # start a timer to figure out how long we're paused for
    pauseTimer = core.Clock()
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='ioHub',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # sleep 1ms so other threads can execute
        clock.time.sleep(0.001)
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # reset any timers
    for timer in timers:
        timer.addTime(-pauseTimer.getTime())


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ioHub'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "WelcomeScreen" ---
    welcome = visual.TextStim(win=win, name='welcome',
        text='welcome to the short term memory experiment.\n\npress the spacebar key to continue.',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    keyWelcome = keyboard.Keyboard(deviceName='keyWelcome')
    
    # --- Initialize components for Routine "BlankScreen" ---
    textBlank = visual.TextStim(win=win, name='textBlank',
        text=None,
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "Instructions" ---
    textinstructions = visual.TextStim(win=win, name='textinstructions',
        text='A series of words will appear, each word lasting for a short viewing time. Try to remember each word that is presented to you.\nPress space bar to continue. ',
        font='Times New Roman',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    keyInstructions = keyboard.Keyboard(deviceName='keyInstructions')
    
    # --- Initialize components for Routine "StudyTrials" ---
    textStudyTrials = visual.TextStim(win=win, name='textStudyTrials',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "BlankScreen" ---
    textBlank = visual.TextStim(win=win, name='textBlank',
        text=None,
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "WaitScreen" ---
    textWaitScreen = visual.TextStim(win=win, name='textWaitScreen',
        text="Please notify the researcher. \nPress '1' to continue",
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    keyWaitScreen = keyboard.Keyboard(deviceName='keyWaitScreen')
    
    # --- Initialize components for Routine "BlankScreen_2" ---
    Blank = visual.TextStim(win=win, name='Blank',
        text=None,
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "Y_N_Screen" ---
    textY_N_Screen = visual.TextStim(win=win, name='textY_N_Screen',
        text="The following words are testing your recall. \nPress either 'Y' for Yes, 'N' for No. ",
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    keyY_N = keyboard.Keyboard(deviceName='keyY_N')
    
    # --- Initialize components for Routine "BlankScreen" ---
    textBlank = visual.TextStim(win=win, name='textBlank',
        text=None,
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "TextTrials" ---
    textTestTrials = visual.TextStim(win=win, name='textTestTrials',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp = keyboard.Keyboard(deviceName='key_resp')
    
    # --- Initialize components for Routine "BlankScreen" ---
    textBlank = visual.TextStim(win=win, name='textBlank',
        text=None,
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "EndScreen" ---
    text = visual.TextStim(win=win, name='text',
        text='Thank you for your participation in this study. \nPlease see the researcher.',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # --- Prepare to start Routine "WelcomeScreen" ---
    # create an object to store info about Routine WelcomeScreen
    WelcomeScreen = data.Routine(
        name='WelcomeScreen',
        components=[welcome, keyWelcome],
    )
    WelcomeScreen.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for keyWelcome
    keyWelcome.keys = []
    keyWelcome.rt = []
    _keyWelcome_allKeys = []
    # store start times for WelcomeScreen
    WelcomeScreen.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    WelcomeScreen.tStart = globalClock.getTime(format='float')
    WelcomeScreen.status = STARTED
    thisExp.addData('WelcomeScreen.started', WelcomeScreen.tStart)
    WelcomeScreen.maxDuration = None
    # keep track of which components have finished
    WelcomeScreenComponents = WelcomeScreen.components
    for thisComponent in WelcomeScreen.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "WelcomeScreen" ---
    WelcomeScreen.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *welcome* updates
        
        # if welcome is starting this frame...
        if welcome.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            welcome.frameNStart = frameN  # exact frame index
            welcome.tStart = t  # local t and not account for scr refresh
            welcome.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(welcome, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'welcome.started')
            # update status
            welcome.status = STARTED
            welcome.setAutoDraw(True)
        
        # if welcome is active this frame...
        if welcome.status == STARTED:
            # update params
            pass
        
        # *keyWelcome* updates
        waitOnFlip = False
        
        # if keyWelcome is starting this frame...
        if keyWelcome.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            keyWelcome.frameNStart = frameN  # exact frame index
            keyWelcome.tStart = t  # local t and not account for scr refresh
            keyWelcome.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(keyWelcome, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'keyWelcome.started')
            # update status
            keyWelcome.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(keyWelcome.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(keyWelcome.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if keyWelcome.status == STARTED and not waitOnFlip:
            theseKeys = keyWelcome.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _keyWelcome_allKeys.extend(theseKeys)
            if len(_keyWelcome_allKeys):
                keyWelcome.keys = _keyWelcome_allKeys[-1].name  # just the last key pressed
                keyWelcome.rt = _keyWelcome_allKeys[-1].rt
                keyWelcome.duration = _keyWelcome_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            WelcomeScreen.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in WelcomeScreen.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "WelcomeScreen" ---
    for thisComponent in WelcomeScreen.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for WelcomeScreen
    WelcomeScreen.tStop = globalClock.getTime(format='float')
    WelcomeScreen.tStopRefresh = tThisFlipGlobal
    thisExp.addData('WelcomeScreen.stopped', WelcomeScreen.tStop)
    # check responses
    if keyWelcome.keys in ['', [], None]:  # No response was made
        keyWelcome.keys = None
    thisExp.addData('keyWelcome.keys',keyWelcome.keys)
    if keyWelcome.keys != None:  # we had a response
        thisExp.addData('keyWelcome.rt', keyWelcome.rt)
        thisExp.addData('keyWelcome.duration', keyWelcome.duration)
    thisExp.nextEntry()
    # the Routine "WelcomeScreen" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "BlankScreen" ---
    # create an object to store info about Routine BlankScreen
    BlankScreen = data.Routine(
        name='BlankScreen',
        components=[textBlank],
    )
    BlankScreen.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for BlankScreen
    BlankScreen.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    BlankScreen.tStart = globalClock.getTime(format='float')
    BlankScreen.status = STARTED
    thisExp.addData('BlankScreen.started', BlankScreen.tStart)
    BlankScreen.maxDuration = None
    # keep track of which components have finished
    BlankScreenComponents = BlankScreen.components
    for thisComponent in BlankScreen.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "BlankScreen" ---
    BlankScreen.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 0.5:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *textBlank* updates
        
        # if textBlank is starting this frame...
        if textBlank.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            textBlank.frameNStart = frameN  # exact frame index
            textBlank.tStart = t  # local t and not account for scr refresh
            textBlank.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(textBlank, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'textBlank.started')
            # update status
            textBlank.status = STARTED
            textBlank.setAutoDraw(True)
        
        # if textBlank is active this frame...
        if textBlank.status == STARTED:
            # update params
            pass
        
        # if textBlank is stopping this frame...
        if textBlank.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > textBlank.tStartRefresh + 0.5-frameTolerance:
                # keep track of stop time/frame for later
                textBlank.tStop = t  # not accounting for scr refresh
                textBlank.tStopRefresh = tThisFlipGlobal  # on global time
                textBlank.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'textBlank.stopped')
                # update status
                textBlank.status = FINISHED
                textBlank.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            BlankScreen.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in BlankScreen.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "BlankScreen" ---
    for thisComponent in BlankScreen.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for BlankScreen
    BlankScreen.tStop = globalClock.getTime(format='float')
    BlankScreen.tStopRefresh = tThisFlipGlobal
    thisExp.addData('BlankScreen.stopped', BlankScreen.tStop)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if BlankScreen.maxDurationReached:
        routineTimer.addTime(-BlankScreen.maxDuration)
    elif BlankScreen.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-0.500000)
    thisExp.nextEntry()
    
    # --- Prepare to start Routine "Instructions" ---
    # create an object to store info about Routine Instructions
    Instructions = data.Routine(
        name='Instructions',
        components=[textinstructions, keyInstructions],
    )
    Instructions.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for keyInstructions
    keyInstructions.keys = []
    keyInstructions.rt = []
    _keyInstructions_allKeys = []
    # store start times for Instructions
    Instructions.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    Instructions.tStart = globalClock.getTime(format='float')
    Instructions.status = STARTED
    thisExp.addData('Instructions.started', Instructions.tStart)
    Instructions.maxDuration = None
    # keep track of which components have finished
    InstructionsComponents = Instructions.components
    for thisComponent in Instructions.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Instructions" ---
    Instructions.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *textinstructions* updates
        
        # if textinstructions is starting this frame...
        if textinstructions.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            textinstructions.frameNStart = frameN  # exact frame index
            textinstructions.tStart = t  # local t and not account for scr refresh
            textinstructions.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(textinstructions, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'textinstructions.started')
            # update status
            textinstructions.status = STARTED
            textinstructions.setAutoDraw(True)
        
        # if textinstructions is active this frame...
        if textinstructions.status == STARTED:
            # update params
            pass
        
        # *keyInstructions* updates
        waitOnFlip = False
        
        # if keyInstructions is starting this frame...
        if keyInstructions.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            keyInstructions.frameNStart = frameN  # exact frame index
            keyInstructions.tStart = t  # local t and not account for scr refresh
            keyInstructions.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(keyInstructions, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'keyInstructions.started')
            # update status
            keyInstructions.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(keyInstructions.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(keyInstructions.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if keyInstructions.status == STARTED and not waitOnFlip:
            theseKeys = keyInstructions.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _keyInstructions_allKeys.extend(theseKeys)
            if len(_keyInstructions_allKeys):
                keyInstructions.keys = _keyInstructions_allKeys[-1].name  # just the last key pressed
                keyInstructions.rt = _keyInstructions_allKeys[-1].rt
                keyInstructions.duration = _keyInstructions_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            Instructions.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in Instructions.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Instructions" ---
    for thisComponent in Instructions.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for Instructions
    Instructions.tStop = globalClock.getTime(format='float')
    Instructions.tStopRefresh = tThisFlipGlobal
    thisExp.addData('Instructions.stopped', Instructions.tStop)
    # check responses
    if keyInstructions.keys in ['', [], None]:  # No response was made
        keyInstructions.keys = None
    thisExp.addData('keyInstructions.keys',keyInstructions.keys)
    if keyInstructions.keys != None:  # we had a response
        thisExp.addData('keyInstructions.rt', keyInstructions.rt)
        thisExp.addData('keyInstructions.duration', keyInstructions.duration)
    thisExp.nextEntry()
    # the Routine "Instructions" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    trials = data.TrialHandler2(
        name='trials',
        nReps=1.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions(
        '/Users/shrutigupta/Downloads/Word_stimuli.xlsx', 
        selection='0:6'
    )
    , 
        seed=None, 
    )
    thisExp.addLoop(trials)  # add the loop to the experiment
    thisTrial = trials.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
    if thisTrial != None:
        for paramName in thisTrial:
            globals()[paramName] = thisTrial[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisTrial in trials:
        currentLoop = trials
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
        if thisTrial != None:
            for paramName in thisTrial:
                globals()[paramName] = thisTrial[paramName]
        
        # --- Prepare to start Routine "StudyTrials" ---
        # create an object to store info about Routine StudyTrials
        StudyTrials = data.Routine(
            name='StudyTrials',
            components=[textStudyTrials],
        )
        StudyTrials.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        textStudyTrials.setColor(WordColor, colorSpace='rgb')
        textStudyTrials.setText(WordItem)
        # store start times for StudyTrials
        StudyTrials.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        StudyTrials.tStart = globalClock.getTime(format='float')
        StudyTrials.status = STARTED
        thisExp.addData('StudyTrials.started', StudyTrials.tStart)
        StudyTrials.maxDuration = None
        # keep track of which components have finished
        StudyTrialsComponents = StudyTrials.components
        for thisComponent in StudyTrials.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "StudyTrials" ---
        # if trial has changed, end Routine now
        if isinstance(trials, data.TrialHandler2) and thisTrial.thisN != trials.thisTrial.thisN:
            continueRoutine = False
        StudyTrials.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 1.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *textStudyTrials* updates
            
            # if textStudyTrials is starting this frame...
            if textStudyTrials.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                textStudyTrials.frameNStart = frameN  # exact frame index
                textStudyTrials.tStart = t  # local t and not account for scr refresh
                textStudyTrials.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(textStudyTrials, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'textStudyTrials.started')
                # update status
                textStudyTrials.status = STARTED
                textStudyTrials.setAutoDraw(True)
            
            # if textStudyTrials is active this frame...
            if textStudyTrials.status == STARTED:
                # update params
                pass
            
            # if textStudyTrials is stopping this frame...
            if textStudyTrials.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > textStudyTrials.tStartRefresh + 1.0-frameTolerance:
                    # keep track of stop time/frame for later
                    textStudyTrials.tStop = t  # not accounting for scr refresh
                    textStudyTrials.tStopRefresh = tThisFlipGlobal  # on global time
                    textStudyTrials.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'textStudyTrials.stopped')
                    # update status
                    textStudyTrials.status = FINISHED
                    textStudyTrials.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                StudyTrials.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in StudyTrials.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "StudyTrials" ---
        for thisComponent in StudyTrials.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for StudyTrials
        StudyTrials.tStop = globalClock.getTime(format='float')
        StudyTrials.tStopRefresh = tThisFlipGlobal
        thisExp.addData('StudyTrials.stopped', StudyTrials.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if StudyTrials.maxDurationReached:
            routineTimer.addTime(-StudyTrials.maxDuration)
        elif StudyTrials.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-1.000000)
        
        # --- Prepare to start Routine "BlankScreen" ---
        # create an object to store info about Routine BlankScreen
        BlankScreen = data.Routine(
            name='BlankScreen',
            components=[textBlank],
        )
        BlankScreen.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for BlankScreen
        BlankScreen.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        BlankScreen.tStart = globalClock.getTime(format='float')
        BlankScreen.status = STARTED
        thisExp.addData('BlankScreen.started', BlankScreen.tStart)
        BlankScreen.maxDuration = None
        # keep track of which components have finished
        BlankScreenComponents = BlankScreen.components
        for thisComponent in BlankScreen.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "BlankScreen" ---
        # if trial has changed, end Routine now
        if isinstance(trials, data.TrialHandler2) and thisTrial.thisN != trials.thisTrial.thisN:
            continueRoutine = False
        BlankScreen.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 0.5:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *textBlank* updates
            
            # if textBlank is starting this frame...
            if textBlank.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                textBlank.frameNStart = frameN  # exact frame index
                textBlank.tStart = t  # local t and not account for scr refresh
                textBlank.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(textBlank, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'textBlank.started')
                # update status
                textBlank.status = STARTED
                textBlank.setAutoDraw(True)
            
            # if textBlank is active this frame...
            if textBlank.status == STARTED:
                # update params
                pass
            
            # if textBlank is stopping this frame...
            if textBlank.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > textBlank.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    textBlank.tStop = t  # not accounting for scr refresh
                    textBlank.tStopRefresh = tThisFlipGlobal  # on global time
                    textBlank.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'textBlank.stopped')
                    # update status
                    textBlank.status = FINISHED
                    textBlank.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                BlankScreen.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in BlankScreen.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "BlankScreen" ---
        for thisComponent in BlankScreen.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for BlankScreen
        BlankScreen.tStop = globalClock.getTime(format='float')
        BlankScreen.tStopRefresh = tThisFlipGlobal
        thisExp.addData('BlankScreen.stopped', BlankScreen.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if BlankScreen.maxDurationReached:
            routineTimer.addTime(-BlankScreen.maxDuration)
        elif BlankScreen.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-0.500000)
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'trials'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "WaitScreen" ---
    # create an object to store info about Routine WaitScreen
    WaitScreen = data.Routine(
        name='WaitScreen',
        components=[textWaitScreen, keyWaitScreen],
    )
    WaitScreen.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for keyWaitScreen
    keyWaitScreen.keys = []
    keyWaitScreen.rt = []
    _keyWaitScreen_allKeys = []
    # store start times for WaitScreen
    WaitScreen.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    WaitScreen.tStart = globalClock.getTime(format='float')
    WaitScreen.status = STARTED
    thisExp.addData('WaitScreen.started', WaitScreen.tStart)
    WaitScreen.maxDuration = None
    # keep track of which components have finished
    WaitScreenComponents = WaitScreen.components
    for thisComponent in WaitScreen.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "WaitScreen" ---
    WaitScreen.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *textWaitScreen* updates
        
        # if textWaitScreen is starting this frame...
        if textWaitScreen.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            textWaitScreen.frameNStart = frameN  # exact frame index
            textWaitScreen.tStart = t  # local t and not account for scr refresh
            textWaitScreen.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(textWaitScreen, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'textWaitScreen.started')
            # update status
            textWaitScreen.status = STARTED
            textWaitScreen.setAutoDraw(True)
        
        # if textWaitScreen is active this frame...
        if textWaitScreen.status == STARTED:
            # update params
            pass
        
        # *keyWaitScreen* updates
        waitOnFlip = False
        
        # if keyWaitScreen is starting this frame...
        if keyWaitScreen.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            keyWaitScreen.frameNStart = frameN  # exact frame index
            keyWaitScreen.tStart = t  # local t and not account for scr refresh
            keyWaitScreen.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(keyWaitScreen, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'keyWaitScreen.started')
            # update status
            keyWaitScreen.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(keyWaitScreen.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(keyWaitScreen.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if keyWaitScreen.status == STARTED and not waitOnFlip:
            theseKeys = keyWaitScreen.getKeys(keyList=['1'], ignoreKeys=["escape"], waitRelease=False)
            _keyWaitScreen_allKeys.extend(theseKeys)
            if len(_keyWaitScreen_allKeys):
                keyWaitScreen.keys = _keyWaitScreen_allKeys[-1].name  # just the last key pressed
                keyWaitScreen.rt = _keyWaitScreen_allKeys[-1].rt
                keyWaitScreen.duration = _keyWaitScreen_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            WaitScreen.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in WaitScreen.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "WaitScreen" ---
    for thisComponent in WaitScreen.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for WaitScreen
    WaitScreen.tStop = globalClock.getTime(format='float')
    WaitScreen.tStopRefresh = tThisFlipGlobal
    thisExp.addData('WaitScreen.stopped', WaitScreen.tStop)
    # check responses
    if keyWaitScreen.keys in ['', [], None]:  # No response was made
        keyWaitScreen.keys = None
    thisExp.addData('keyWaitScreen.keys',keyWaitScreen.keys)
    if keyWaitScreen.keys != None:  # we had a response
        thisExp.addData('keyWaitScreen.rt', keyWaitScreen.rt)
        thisExp.addData('keyWaitScreen.duration', keyWaitScreen.duration)
    thisExp.nextEntry()
    # the Routine "WaitScreen" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "BlankScreen_2" ---
    # create an object to store info about Routine BlankScreen_2
    BlankScreen_2 = data.Routine(
        name='BlankScreen_2',
        components=[Blank],
    )
    BlankScreen_2.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for BlankScreen_2
    BlankScreen_2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    BlankScreen_2.tStart = globalClock.getTime(format='float')
    BlankScreen_2.status = STARTED
    thisExp.addData('BlankScreen_2.started', BlankScreen_2.tStart)
    BlankScreen_2.maxDuration = None
    # keep track of which components have finished
    BlankScreen_2Components = BlankScreen_2.components
    for thisComponent in BlankScreen_2.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "BlankScreen_2" ---
    BlankScreen_2.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 0.5:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *Blank* updates
        
        # if Blank is starting this frame...
        if Blank.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            Blank.frameNStart = frameN  # exact frame index
            Blank.tStart = t  # local t and not account for scr refresh
            Blank.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Blank, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'Blank.started')
            # update status
            Blank.status = STARTED
            Blank.setAutoDraw(True)
        
        # if Blank is active this frame...
        if Blank.status == STARTED:
            # update params
            pass
        
        # if Blank is stopping this frame...
        if Blank.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > Blank.tStartRefresh + 0.5-frameTolerance:
                # keep track of stop time/frame for later
                Blank.tStop = t  # not accounting for scr refresh
                Blank.tStopRefresh = tThisFlipGlobal  # on global time
                Blank.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Blank.stopped')
                # update status
                Blank.status = FINISHED
                Blank.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            BlankScreen_2.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in BlankScreen_2.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "BlankScreen_2" ---
    for thisComponent in BlankScreen_2.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for BlankScreen_2
    BlankScreen_2.tStop = globalClock.getTime(format='float')
    BlankScreen_2.tStopRefresh = tThisFlipGlobal
    thisExp.addData('BlankScreen_2.stopped', BlankScreen_2.tStop)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if BlankScreen_2.maxDurationReached:
        routineTimer.addTime(-BlankScreen_2.maxDuration)
    elif BlankScreen_2.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-0.500000)
    thisExp.nextEntry()
    
    # --- Prepare to start Routine "Y_N_Screen" ---
    # create an object to store info about Routine Y_N_Screen
    Y_N_Screen = data.Routine(
        name='Y_N_Screen',
        components=[textY_N_Screen, keyY_N],
    )
    Y_N_Screen.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for keyY_N
    keyY_N.keys = []
    keyY_N.rt = []
    _keyY_N_allKeys = []
    # store start times for Y_N_Screen
    Y_N_Screen.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    Y_N_Screen.tStart = globalClock.getTime(format='float')
    Y_N_Screen.status = STARTED
    thisExp.addData('Y_N_Screen.started', Y_N_Screen.tStart)
    Y_N_Screen.maxDuration = None
    # keep track of which components have finished
    Y_N_ScreenComponents = Y_N_Screen.components
    for thisComponent in Y_N_Screen.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Y_N_Screen" ---
    Y_N_Screen.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *textY_N_Screen* updates
        
        # if textY_N_Screen is starting this frame...
        if textY_N_Screen.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            textY_N_Screen.frameNStart = frameN  # exact frame index
            textY_N_Screen.tStart = t  # local t and not account for scr refresh
            textY_N_Screen.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(textY_N_Screen, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'textY_N_Screen.started')
            # update status
            textY_N_Screen.status = STARTED
            textY_N_Screen.setAutoDraw(True)
        
        # if textY_N_Screen is active this frame...
        if textY_N_Screen.status == STARTED:
            # update params
            pass
        
        # *keyY_N* updates
        waitOnFlip = False
        
        # if keyY_N is starting this frame...
        if keyY_N.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            keyY_N.frameNStart = frameN  # exact frame index
            keyY_N.tStart = t  # local t and not account for scr refresh
            keyY_N.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(keyY_N, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'keyY_N.started')
            # update status
            keyY_N.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(keyY_N.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(keyY_N.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if keyY_N.status == STARTED and not waitOnFlip:
            theseKeys = keyY_N.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _keyY_N_allKeys.extend(theseKeys)
            if len(_keyY_N_allKeys):
                keyY_N.keys = _keyY_N_allKeys[-1].name  # just the last key pressed
                keyY_N.rt = _keyY_N_allKeys[-1].rt
                keyY_N.duration = _keyY_N_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            Y_N_Screen.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in Y_N_Screen.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Y_N_Screen" ---
    for thisComponent in Y_N_Screen.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for Y_N_Screen
    Y_N_Screen.tStop = globalClock.getTime(format='float')
    Y_N_Screen.tStopRefresh = tThisFlipGlobal
    thisExp.addData('Y_N_Screen.stopped', Y_N_Screen.tStop)
    # check responses
    if keyY_N.keys in ['', [], None]:  # No response was made
        keyY_N.keys = None
    thisExp.addData('keyY_N.keys',keyY_N.keys)
    if keyY_N.keys != None:  # we had a response
        thisExp.addData('keyY_N.rt', keyY_N.rt)
        thisExp.addData('keyY_N.duration', keyY_N.duration)
    thisExp.nextEntry()
    # the Routine "Y_N_Screen" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "BlankScreen" ---
    # create an object to store info about Routine BlankScreen
    BlankScreen = data.Routine(
        name='BlankScreen',
        components=[textBlank],
    )
    BlankScreen.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for BlankScreen
    BlankScreen.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    BlankScreen.tStart = globalClock.getTime(format='float')
    BlankScreen.status = STARTED
    thisExp.addData('BlankScreen.started', BlankScreen.tStart)
    BlankScreen.maxDuration = None
    # keep track of which components have finished
    BlankScreenComponents = BlankScreen.components
    for thisComponent in BlankScreen.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "BlankScreen" ---
    BlankScreen.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 0.5:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *textBlank* updates
        
        # if textBlank is starting this frame...
        if textBlank.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            textBlank.frameNStart = frameN  # exact frame index
            textBlank.tStart = t  # local t and not account for scr refresh
            textBlank.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(textBlank, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'textBlank.started')
            # update status
            textBlank.status = STARTED
            textBlank.setAutoDraw(True)
        
        # if textBlank is active this frame...
        if textBlank.status == STARTED:
            # update params
            pass
        
        # if textBlank is stopping this frame...
        if textBlank.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > textBlank.tStartRefresh + 0.5-frameTolerance:
                # keep track of stop time/frame for later
                textBlank.tStop = t  # not accounting for scr refresh
                textBlank.tStopRefresh = tThisFlipGlobal  # on global time
                textBlank.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'textBlank.stopped')
                # update status
                textBlank.status = FINISHED
                textBlank.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            BlankScreen.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in BlankScreen.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "BlankScreen" ---
    for thisComponent in BlankScreen.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for BlankScreen
    BlankScreen.tStop = globalClock.getTime(format='float')
    BlankScreen.tStopRefresh = tThisFlipGlobal
    thisExp.addData('BlankScreen.stopped', BlankScreen.tStop)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if BlankScreen.maxDurationReached:
        routineTimer.addTime(-BlankScreen.maxDuration)
    elif BlankScreen.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-0.500000)
    thisExp.nextEntry()
    
    # set up handler to look after randomisation of conditions etc
    trials_2 = data.TrialHandler2(
        name='trials_2',
        nReps=1.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('/Users/shrutigupta/Downloads/Word_stimuli.xlsx'), 
        seed=None, 
    )
    thisExp.addLoop(trials_2)  # add the loop to the experiment
    thisTrial_2 = trials_2.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrial_2.rgb)
    if thisTrial_2 != None:
        for paramName in thisTrial_2:
            globals()[paramName] = thisTrial_2[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisTrial_2 in trials_2:
        currentLoop = trials_2
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisTrial_2.rgb)
        if thisTrial_2 != None:
            for paramName in thisTrial_2:
                globals()[paramName] = thisTrial_2[paramName]
        
        # --- Prepare to start Routine "TextTrials" ---
        # create an object to store info about Routine TextTrials
        TextTrials = data.Routine(
            name='TextTrials',
            components=[textTestTrials, key_resp],
        )
        TextTrials.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        textTestTrials.setText(WordItem)
        # create starting attributes for key_resp
        key_resp.keys = []
        key_resp.rt = []
        _key_resp_allKeys = []
        # store start times for TextTrials
        TextTrials.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        TextTrials.tStart = globalClock.getTime(format='float')
        TextTrials.status = STARTED
        thisExp.addData('TextTrials.started', TextTrials.tStart)
        TextTrials.maxDuration = None
        # keep track of which components have finished
        TextTrialsComponents = TextTrials.components
        for thisComponent in TextTrials.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "TextTrials" ---
        # if trial has changed, end Routine now
        if isinstance(trials_2, data.TrialHandler2) and thisTrial_2.thisN != trials_2.thisTrial.thisN:
            continueRoutine = False
        TextTrials.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *textTestTrials* updates
            
            # if textTestTrials is starting this frame...
            if textTestTrials.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                textTestTrials.frameNStart = frameN  # exact frame index
                textTestTrials.tStart = t  # local t and not account for scr refresh
                textTestTrials.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(textTestTrials, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'textTestTrials.started')
                # update status
                textTestTrials.status = STARTED
                textTestTrials.setAutoDraw(True)
            
            # if textTestTrials is active this frame...
            if textTestTrials.status == STARTED:
                # update params
                pass
            
            # *key_resp* updates
            waitOnFlip = False
            
            # if key_resp is starting this frame...
            if key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_resp.frameNStart = frameN  # exact frame index
                key_resp.tStart = t  # local t and not account for scr refresh
                key_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp.started')
                # update status
                key_resp.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp.status == STARTED and not waitOnFlip:
                theseKeys = key_resp.getKeys(keyList=['y','n'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_allKeys.extend(theseKeys)
                if len(_key_resp_allKeys):
                    key_resp.keys = _key_resp_allKeys[-1].name  # just the last key pressed
                    key_resp.rt = _key_resp_allKeys[-1].rt
                    key_resp.duration = _key_resp_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                TextTrials.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in TextTrials.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "TextTrials" ---
        for thisComponent in TextTrials.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for TextTrials
        TextTrials.tStop = globalClock.getTime(format='float')
        TextTrials.tStopRefresh = tThisFlipGlobal
        thisExp.addData('TextTrials.stopped', TextTrials.tStop)
        # check responses
        if key_resp.keys in ['', [], None]:  # No response was made
            key_resp.keys = None
        trials_2.addData('key_resp.keys',key_resp.keys)
        if key_resp.keys != None:  # we had a response
            trials_2.addData('key_resp.rt', key_resp.rt)
            trials_2.addData('key_resp.duration', key_resp.duration)
        # the Routine "TextTrials" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "BlankScreen" ---
        # create an object to store info about Routine BlankScreen
        BlankScreen = data.Routine(
            name='BlankScreen',
            components=[textBlank],
        )
        BlankScreen.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for BlankScreen
        BlankScreen.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        BlankScreen.tStart = globalClock.getTime(format='float')
        BlankScreen.status = STARTED
        thisExp.addData('BlankScreen.started', BlankScreen.tStart)
        BlankScreen.maxDuration = None
        # keep track of which components have finished
        BlankScreenComponents = BlankScreen.components
        for thisComponent in BlankScreen.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "BlankScreen" ---
        # if trial has changed, end Routine now
        if isinstance(trials_2, data.TrialHandler2) and thisTrial_2.thisN != trials_2.thisTrial.thisN:
            continueRoutine = False
        BlankScreen.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 0.5:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *textBlank* updates
            
            # if textBlank is starting this frame...
            if textBlank.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                textBlank.frameNStart = frameN  # exact frame index
                textBlank.tStart = t  # local t and not account for scr refresh
                textBlank.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(textBlank, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'textBlank.started')
                # update status
                textBlank.status = STARTED
                textBlank.setAutoDraw(True)
            
            # if textBlank is active this frame...
            if textBlank.status == STARTED:
                # update params
                pass
            
            # if textBlank is stopping this frame...
            if textBlank.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > textBlank.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    textBlank.tStop = t  # not accounting for scr refresh
                    textBlank.tStopRefresh = tThisFlipGlobal  # on global time
                    textBlank.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'textBlank.stopped')
                    # update status
                    textBlank.status = FINISHED
                    textBlank.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                BlankScreen.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in BlankScreen.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "BlankScreen" ---
        for thisComponent in BlankScreen.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for BlankScreen
        BlankScreen.tStop = globalClock.getTime(format='float')
        BlankScreen.tStopRefresh = tThisFlipGlobal
        thisExp.addData('BlankScreen.stopped', BlankScreen.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if BlankScreen.maxDurationReached:
            routineTimer.addTime(-BlankScreen.maxDuration)
        elif BlankScreen.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-0.500000)
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'trials_2'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "EndScreen" ---
    # create an object to store info about Routine EndScreen
    EndScreen = data.Routine(
        name='EndScreen',
        components=[text],
    )
    EndScreen.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for EndScreen
    EndScreen.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    EndScreen.tStart = globalClock.getTime(format='float')
    EndScreen.status = STARTED
    thisExp.addData('EndScreen.started', EndScreen.tStart)
    EndScreen.maxDuration = None
    # keep track of which components have finished
    EndScreenComponents = EndScreen.components
    for thisComponent in EndScreen.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "EndScreen" ---
    EndScreen.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 5.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text* updates
        
        # if text is starting this frame...
        if text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text.frameNStart = frameN  # exact frame index
            text.tStart = t  # local t and not account for scr refresh
            text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text.started')
            # update status
            text.status = STARTED
            text.setAutoDraw(True)
        
        # if text is active this frame...
        if text.status == STARTED:
            # update params
            pass
        
        # if text is stopping this frame...
        if text.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text.tStartRefresh + 5-frameTolerance:
                # keep track of stop time/frame for later
                text.tStop = t  # not accounting for scr refresh
                text.tStopRefresh = tThisFlipGlobal  # on global time
                text.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text.stopped')
                # update status
                text.status = FINISHED
                text.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            EndScreen.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in EndScreen.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "EndScreen" ---
    for thisComponent in EndScreen.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for EndScreen
    EndScreen.tStop = globalClock.getTime(format='float')
    EndScreen.tStopRefresh = tThisFlipGlobal
    thisExp.addData('EndScreen.stopped', EndScreen.tStop)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if EndScreen.maxDurationReached:
        routineTimer.addTime(-EndScreen.maxDuration)
    elif EndScreen.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-5.000000)
    thisExp.nextEntry()
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # return console logger level to WARNING
    logging.console.setLevel(logging.WARNING)
    # mark experiment handler as finished
    thisExp.status = FINISHED
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
