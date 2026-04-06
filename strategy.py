#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Donchian Breakout with Volume Confirmation and ATR Stop.
# Uses 4h Donchian(20) channels for breakout signals in both directions.
# Volume filter: current volume > 1.5x 20-period average to filter weak breakouts.
# Stoploss: 2x ATR(14) from entry, with position sizing of 0.25.
# Works in bull/bear markets via breakout logic and volatility-based stops.
# Target: 75-200 trades over 4 years (19-50/year).

name = "4h_donchian20_volume_atr_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period high/low)
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    for i in range(19, n):
        highest_high[i] = np.max(high[i-19:i+1])
        lowest_low[i] = np.min(low[i-19:i+1])
    
    # ATR(14) for stoploss
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]  # first bar
    atr = np.full(n, np.nan)
    for i in range(13, n):
        atr[i] = np.mean(tr[i-13:i+1])
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if ATR or volume MA not ready
        if np.isnan(atr[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        volume_filter = volume[i] > vol_ma[i] * 1.5
        
        # Check exits and stoploss
        if position == 1:  # long position
            stop_loss_level = entry_price - 2.0 * atr[i]
            if close[i] < stop_loss_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            stop_loss_level = entry_price + 2.0 * atr[i]
            if close[i] > stop_loss_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation
            if volume_filter:
                # Long breakout above upper band
                if close[i] > highest_high[i] and close[i-1] <= highest_high[i]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short breakdown below lower band
                elif close[i] < lowest_low[i] and close[i-1] >= lowest_low[i]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
    
    return signals