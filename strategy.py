#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA34 trend filter and volume confirmation
# Donchian breakouts capture strong momentum moves. 1w EMA34 ensures alignment with weekly trend.
# Volume spike (>1.5x 20 EMA) confirms institutional participation. Discrete sizing 0.25 limits risk.
# Target: 30-80 trades over 4 years (7-20/year). Works in bull/bear via trend filter.

name = "1d_Donchian20_1wEMA34_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 1w EMA34 for trend direction
    close_1w = pd.Series(df_1w['close'])
    ema34_1w = close_1w.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1w EMA34 to 1d timeframe (completed 1w bar only)
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Calculate Donchian channels on 1d timeframe (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max()
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min()
    
    # Volume confirmation: 20-period EMA of volume on 1d timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema34_1w_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.5 x 20-period EMA
        volume_confirm = volume[i] > (1.5 * vol_ema_20[i])
        
        if position == 0:
            # Long conditions: price breaks above upper Donchian + uptrend + volume spike
            if close[i] > highest_high[i] and close[i] > ema34_1w_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below lower Donchian + downtrend + volume spike
            elif close[i] < lowest_low[i] and close[i] < ema34_1w_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to midpoint OR trend changes OR volume drops
            midpoint = (highest_high[i] + lowest_low[i]) / 2
            if (close[i] < midpoint or 
                close[i] < ema34_1w_aligned[i] or 
                volume[i] < vol_ema_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to midpoint OR trend changes OR volume drops
            midpoint = (highest_high[i] + lowest_low[i]) / 2
            if (close[i] > midpoint or 
                close[i] > ema34_1w_aligned[i] or 
                volume[i] < vol_ema_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals