#!/usr/bin/env python3
import numpy as np
import pandas as pd

name = "BTC Swinger v1"
timeframe = "1d"
leverage = 1

ATR_LENGTH = 3
ATR_MULT = 1.0
STRATEGY_DIRECTION = 1  # 1: Long, -1: Short, 0: All

def generate_signals(prices):
    """
    Generates trading signals based on BTC Swinger v1 logic.
    Args:
        prices (pd.DataFrame): Must contain 'open_time', 'open', 'high', 'low', 'close', 'volume'.
    Returns:
        np.ndarray: Array of signals (1: Long, -1: Short, 0: Neutral) matching len(prices).
    """
    n = len(prices)
    signals = np.zeros(n, dtype=int)
    
    if n < ATR_LENGTH + 1:
        return signals
        
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate True Range
    tr = np.zeros(n)
    tr[1:] = np.maximum(
        high[1:] - low[1:],
        np.maximum(
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
    )
    tr[0] = high[0] - low[0]
    
    # Calculate ATR (SMA of TR)
    atr = np.zeros(n)
    atr_sum = np.sum(tr[:ATR_LENGTH])
    atr[ATR_LENGTH-1] = atr_sum / ATR_LENGTH
    for i in range(ATR_LENGTH, n):
        atr[i] = (atr[i-1] * (ATR_LENGTH - 1) + tr[i]) / ATR_LENGTH
        
    # Stateful VStop Calculation (Chandelier Exit logic)
    vstop = np.zeros(n)
    is_uptrend = np.ones(n, dtype=bool)
    max_val = np.zeros(n)
    min_val = np.zeros(n)
    
    # Initialize state at first valid ATR index to avoid NaN propagation
    start_idx = ATR_LENGTH
    max_val[start_idx] = close[start_idx]
    min_val[start_idx] = close[start_idx]
    is_uptrend[start_idx] = True
    vstop[start_idx] = max_val[start_idx] - ATR_MULT * atr[start_idx]
    
    for i in range(start_idx + 1, n):
        max_prev = max_val[i-1]
        min_prev = min_val[i-1]
        is_uptrend_prev = is_uptrend[i-1]
        vstop_prev = vstop[i-1]
        atr_curr = atr[i]
        close_curr = close[i]
        
        max1 = max(max_prev, close_curr)
        min1 = min(min_prev, close_curr)
        
        stop_val = (max1 - ATR_MULT * atr_curr) if is_uptrend_prev else (min1 + ATR_MULT * atr_curr)
        vstop1 = max(vstop_prev, stop_val) if is_uptrend_prev else min(vstop_prev, stop_val)
        
        is_uptrend_curr = (close_curr - vstop1) >= 0
        is_trend_changed = is_uptrend_curr != is_uptrend_prev
        
        if is_trend_changed:
            max_val[i] = close_curr
            min_val[i] = close_curr
            vstop[i] = (max_val[i] - ATR_MULT * atr_curr) if is_uptrend_curr else (min_val[i] + ATR_MULT * atr_curr)
        else:
            max_val[i] = max1
            min_val[i] = min1
            vstop[i] = vstop1
            
        is_uptrend[i] = is_uptrend_curr
        
    # Generate Signals
    # Date range filtering handled by backtest config per adaptation notes
    for i in range(start_idx, n):
        if atr[i] <= 0:
            continue
            
        if STRATEGY_DIRECTION == 1:
            if close[i] > vstop[i]:
                signals[i] = 1
        elif STRATEGY_DIRECTION == -1:
            if close[i] < vstop[i]:
                signals[i] = -1
        else:
            if close[i] > vstop[i]:
                signals[i] = 1
            elif close[i] < vstop[i]:
                signals[i] = -1
                
    return signals