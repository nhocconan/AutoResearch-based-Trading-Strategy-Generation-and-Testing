#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian Breakout with Volume Confirmation and ATR Stop
# Uses 4h Donchian channel breakout (20-period) confirmed by volume spike and ATR-based stop loss.
# Works in bull markets (breakouts up) and bear markets (breakouts down). Target: 50-150 total trades.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channel (20-period high/low)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate ATR (14-period) for stop loss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation: current volume > 1.5x 20-period median volume
    vol_median = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(atr[i]) or np.isnan(vol_median[i])):
            continue
        
        # Long entry: price breaks above Donchian high + volume confirmation
        if (close[i] > donchian_high[i] and
            volume[i] > 1.5 * vol_median[i] and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: price breaks below Donchian low + volume confirmation
        elif (close[i] < donchian_low[i] and
              volume[i] > 1.5 * vol_median[i] and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: ATR-based stop loss
        elif position == 1 and close[i] < donchian_high[i] - 2.0 * atr[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > donchian_low[i] + 2.0 * atr[i]:
            position = 0
            signals[i] = 0.0
    
    return signals

name = "4h_Donchian_Breakout_Volume_ATR"
timeframe = "4h"
leverage = 1.0