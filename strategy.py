#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_13971_6h_atr_breakout_volume_v1"
timeframe = "6h"
leverage = 1.0

# Hypothesis: 6h ATR breakout with volume confirmation and ATR-based stop loss.
# Uses ATR(20) as dynamic threshold for breakouts: long when price > close[0] + 1.5*ATR,
# short when price < close[0] - 1.5*ATR, where close[0] is the open of the current bar.
# Volume must be > 1.5x 20-period average for confirmation.
# Stop loss at 2*ATR from entry. Target: 75-150 total trades over 4 years (19-38/year).
# Works in bull (breakouts with volume) and bear (breakdowns with volume).

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # 6h data for breakout detection, ATR, and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # ATR for breakout threshold and stop loss
    atr = calculate_atr(high, low, close, 20)
    
    # Volume confirmation
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(50, 20) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(atr[i]) or np.isnan(volume_ma[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check stops
        if position == 1:  # long position
            # Check stop loss
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # short position
            # Check stop loss
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * 1.5)
        
        # Breakout signals using ATR threshold
        breakout_long = close[i] > (high[i] + atr[i] * 1.5)
        breakout_short = close[i] < (low[i] - atr[i] * 1.5)
        
        # Entry signals
        long_signal = breakout_long and volume_ok
        short_signal = breakout_short and volume_ok
        
        # Generate signals
        if position == 0:
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (2.0 * atr[i])
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (2.0 * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long position
            signals[i] = 0.25
        elif position == -1:
            # Hold short position
            signals[i] = -0.25
    
    return signals