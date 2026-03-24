#!/usr/bin/env python3
import numpy as np
import pandas as pd

name = "BTCbot"
timeframe = "1h"
leverage = 1

def wma(data, length):
    """Calculate Weighted Moving Average using numpy."""
    weights = np.arange(1, length + 1)
    if len(data) < length:
        return np.full_like(data, np.nan, dtype=float)
    conv = np.convolve(data, weights, mode='valid')
    pad = np.full(length - 1, np.nan)
    return np.concatenate([pad, conv / weights.sum()])

def generate_signals(prices):
    """
    Generate trading signals based on adapted BTCbot logic.
    
    Args:
        prices (pd.DataFrame): DataFrame with columns ['open_time', 'open', 'high', 'low', 'close', 'volume'].
        
    Returns:
        np.ndarray: Array of signals (1.0 for Long, -1.0 for Short, 0.0 for Neutral).
    """
    n = len(prices)
    signals = np.zeros(n)
    
    # Extract numpy arrays
    o = prices['open'].values
    h = prices['high'].values
    l = prices['low'].values
    c = prices['close'].values
    
    # Calculate OHLC4
    ohlc4 = (o + h + l + c) / 4.0
    
    # src = ohlc4[1] (Previous bar's OHLC4)
    src = np.empty_like(ohlc4)
    src[1:] = ohlc4[:-1]
    src[0] = np.nan
    
    # Parameters
    keh = 7
    wma_short = int(round(keh / 2))  # 4
    wma_long = keh                   # 7
    esqn = int(round(np.sqrt(keh)))  # 3
    dt = 0.001
    
    # Note: External symbols (DXY, XAUAUD) are unavailable in this interface.
    # We substitute them with the current symbol's data (BTC) to preserve logic structure.
    # This is a partial adaptation as per metadata constraints.
    
    # Calculate s1, s2 (BTC Component)
    s2ma = 2 * wma(src, wma_short)
    sma = wma(src, wma_long)
    sdiff = s2ma - sma
    s1 = wma(sdiff, esqn)
    
    # Calculate s2 (using src[1] logic)
    src_prev = np.empty_like(src)
    src_prev[1:] = src[:-1]
    src_prev[0] = np.nan
    
    s2ma1 = 2 * wma(src_prev, wma_short)
    sma1 = wma(src_prev, wma_long)
    sdiff1 = s2ma1 - sma1
    s2 = wma(sdiff1, esqn)
    
    # Substitute n1, n2 (XAU Component) with BTC logic due to data constraints
    n1 = s1.copy()
    n2 = s2.copy()
    
    # Substitute e1, e2 (DXY Component) with BTC logic due to data constraints
    e1 = s1.copy()
    e2 = s2.copy()
    
    # Confidence: Approximate Daily ROC using 24 periods (assuming 1h timeframe)
    # Pine: (security(tickerid, 'D', src) - security(tickerid, 'D', src[1])) / security(...)
    confidence = np.full(n, np.nan)
    for i in range(24, n):
        prev_val = src[i - 24]
        if prev_val != 0 and not np.isnan(prev_val):
            confidence[i] = (src[i] - prev_val) / prev_val
    
    # State management for pyramiding=0 (max 1 open trade)
    position = 0  # 0: None, 1: Long, -1: Short
    
    for i in range(n):
        # Check for NaNs in indicators
        if np.isnan(e1[i]) or np.isnan(e2[i]) or np.isnan(n1[i]) or np.isnan(n2[i]) or \
           np.isnan(s1[i]) or np.isnan(s2[i]) or np.isnan(confidence[i]):
            signals[i] = position
            continue
        
        # Exit Conditions
        closelong = (e1[i] > e2[i]) and (confidence[i] < dt)
        closeshort = (e1[i] < e2[i]) and (confidence[i] > dt)
        
        # Entry Conditions
        longCondition = (e1[i] < e2[i]) and (n1[i] > n2[i]) and (s1[i] > s2[i]) and (confidence[i] > dt)
        shortCondition = (e1[i] > e2[i]) and (n1[i] < n2[i]) and (s1[i] < s2[i]) and (confidence[i] < dt)
        
        if position == 1:
            if closelong:
                position = 0
        elif position == -1:
            if closeshort:
                position = 0
        else:
            if longCondition:
                position = 1
            elif shortCondition:
                position = -1
        
        signals[i] = position
        
    return signals
