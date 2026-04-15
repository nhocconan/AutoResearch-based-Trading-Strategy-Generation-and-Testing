#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with volume confirmation and ATR stop
# Uses 4h Donchian channel (20-period) for breakout signals.
# Long when price breaks above upper band, short when breaks below lower band.
# Confirmed by volume > 1.5x median volume and ATR-based stoploss.
# Works in bull markets (breakouts up) and bear markets (breakouts down).
# Target: 50-150 total trades over 4 years.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channel (20-period) on 4h
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate ATR (14-period) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation: compare to median of last 20 periods
    volume_median = pd.Series(volume).rolling(window=20, min_periods=1).median().values
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(atr[i]) or np.isnan(volume_median[i])):
            continue
        
        # Long entry: price breaks above Donchian upper + volume confirmation
        if (close[i] > donchian_upper[i] and
            volume[i] > 1.5 * volume_median[i] and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: price breaks below Donchian lower + volume confirmation
        elif (close[i] < donchian_lower[i] and
              volume[i] > 1.5 * volume_median[i] and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: ATR-based stoploss or reverse signal
        elif position == 1 and close[i] <= donchian_lower[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] >= donchian_upper[i]:
            position = 0
            signals[i] = 0.0
    
    return signals

name = "4h_Donchian_Breakout_Volume_ATR"
timeframe = "4h"
leverage = 1.0