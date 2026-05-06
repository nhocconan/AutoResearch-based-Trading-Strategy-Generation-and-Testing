#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and ATR trailing stop
# Uses price channel breakouts as primary signal, volume > 1.5x 20-bar average for confirmation
# ATR-based trailing stop (2.5x ATR) to manage risk and reduce whipsaw
# Discrete sizing 0.25 to limit fee drag; target 80-180 total trades over 4 years (20-45/year)
# Works in both bull/bear markets: breakouts capture momentum, volume filter avoids false signals

name = "4h_Donchian20_VolumeConfirm_ATRStop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate volume confirmation (>1.5x 20-bar average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    # Calculate ATR(14) for trailing stop
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_high_since_entry = 0.0
    lowest_low_since_entry = 0.0
    
    for i in range(20, n):
        # Skip if any critical value is NaN
        if np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or np.isnan(volume_filter[i]) or np.isnan(atr[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price > upper Donchian AND volume confirmation
            if close[i] > high_roll[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
                highest_high_since_entry = high[i]
            # Short breakdown: price < lower Donchian AND volume confirmation
            elif close[i] < low_roll[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
                lowest_low_since_entry = low[i]
        elif position == 1:
            # Update highest high since entry
            if high[i] > highest_high_since_entry:
                highest_high_since_entry = high[i]
            # ATR trailing stop: exit if price drops 2.5*ATR from highest high
            if close[i] < highest_high_since_entry - (2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Update lowest low since entry
            if low[i] < lowest_low_since_entry:
                lowest_low_since_entry = low[i]
            # ATR trailing stop: exit if price rises 2.5*ATR from lowest low
            if close[i] > lowest_low_since_entry + (2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals