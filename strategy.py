#!/usr/bin/env python3
# Hypothesis: 4h Donchian(20) breakout with volume confirmation and ATR trailing stop.
# Long when price breaks above 20-period high AND volume > 1.5x average volume.
# Short when price breaks below 20-period low AND volume > 1.5x average volume.
# Exit on ATR(14) trailing stop (2.5x) or opposite Donchian breakout.
# Uses discrete sizing 0.25 to minimize fee churn. Target: 75-200 trades over 4 years.
# Donchian channels provide objective structure, volume confirms breakout strength,
# ATR stop manages risk without look-ahead. Works in bull/bear via symmetric logic.

name = "4h_Donchian20_VolumeBreakout_ATRStop_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate ATR(14) for trailing stop
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar has no previous close
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate average volume for confirmation
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = np.full(n, np.nan)  # Track highest high since entry for longs
    lowest_since_entry = np.full(n, np.nan)   # Track lowest low since entry for shorts
    
    for i in range(20, n):  # Start after sufficient data for Donchian
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(atr[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above 20-period high AND volume > 1.5x average
            if close[i] > highest_high[i] and volume[i] > 1.5 * avg_volume[i]:
                signals[i] = 0.25
                position = 1
                highest_since_entry[i] = high[i]  # Initialize tracking
            # SHORT: Price breaks below 20-period low AND volume > 1.5x average
            elif close[i] < lowest_low[i] and volume[i] > 1.5 * avg_volume[i]:
                signals[i] = -0.25
                position = -1
                lowest_since_entry[i] = low[i]  # Initialize tracking
            else:
                signals[i] = 0.0
                # Carry forward tracking values when flat
                if i > 0:
                    highest_since_entry[i] = highest_since_entry[i-1]
                    lowest_since_entry[i] = lowest_since_entry[i-1]
        elif position == 1:
            # Update highest high since entry
            highest_since_entry[i] = max(highest_since_entry[i-1], high[i])
            # EXIT LONG: trailing stop hit (2.5x ATR) OR opposite breakout
            trailing_stop = close[i] < (highest_since_entry[i] - 2.5 * atr[i])
            opposite_break = close[i] < lowest_low[i]  # Break below Donchian low
            if trailing_stop or opposite_break:
                signals[i] = 0.0
                position = 0
                # Reset tracking when flat
                highest_since_entry[i] = np.nan
            else:
                signals[i] = 0.25
                # Carry forward tracking
                if i > 0:
                    highest_since_entry[i] = highest_since_entry[i-1]
        elif position == -1:
            # Update lowest low since entry
            lowest_since_entry[i] = min(lowest_since_entry[i-1], low[i])
            # EXIT SHORT: trailing stop hit (2.5x ATR) OR opposite breakout
            trailing_stop = close[i] > (lowest_since_entry[i] + 2.5 * atr[i])
            opposite_break = close[i] > highest_high[i]  # Break above Donchian high
            if trailing_stop or opposite_break:
                signals[i] = 0.0
                position = 0
                # Reset tracking when flat
                lowest_since_entry[i] = np.nan
            else:
                signals[i] = -0.25
                # Carry forward tracking
                if i > 0:
                    lowest_since_entry[i] = lowest_since_entry[i-1]
    
    return signals