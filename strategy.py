#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation (>1.5x 20-period average)
# Donchian channels capture significant breakouts with clear structure. 12h EMA50 ensures alignment with higher timeframe trend.
# Volume spike filter confirms institutional participation, reducing false breakouts.
# Discrete position sizing (0.25) minimizes fee churn while maintaining meaningful exposure.
# Target: 75-150 total trades over 4 years (19-37/year) on 4h timeframe.

name = "4h_Donchian20_12hEMA50_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 1:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 20-period Donchian channels on 4h timeframe
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 20-period average volume for spike confirmation (on 4h timeframe)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # 12h EMA50, Donchian warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(highest_20[i]) or 
            np.isnan(lowest_20[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema_12h = ema_50_12h_aligned[i]
        curr_highest_20 = highest_20[i]
        curr_lowest_20 = lowest_20[i]
        curr_vol_ma = vol_ma_20[i]
        curr_volume = volume[i]
        
        # Volume spike confirmation: current volume > 1.5x 20-period average
        vol_spike = curr_volume > 1.5 * curr_vol_ma
        
        # Donchian breakout conditions
        breakout_long = curr_high > curr_highest_20   # price breaks above upper channel
        breakout_short = curr_low < curr_lowest_20    # price breaks below lower channel
        
        # Handle exits
        if position == 1:  # Long position
            # Exit: price retracement to midpoint OR trend turns bearish
            midpoint = (curr_highest_20 + curr_lowest_20) / 2
            if (curr_close < midpoint or curr_close < curr_ema_12h):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price retracement to midpoint OR trend turns bullish
            midpoint = (curr_highest_20 + curr_lowest_20) / 2
            if (curr_close > midpoint or curr_close > curr_ema_12h):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: bullish breakout above upper channel AND above 12h EMA50 AND volume spike
            if (breakout_long and 
                curr_close > curr_ema_12h and 
                vol_spike):
                signals[i] = 0.25
                position = 1
            # Short entry: bearish breakout below lower channel AND below 12h EMA50 AND volume spike
            elif (breakout_short and 
                  curr_close < curr_ema_12h and 
                  vol_spike):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals