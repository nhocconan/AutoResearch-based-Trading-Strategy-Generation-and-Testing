#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and volume spike (>1.5x 20-period average)
# Donchian breakouts capture strong momentum moves; 1d EMA50 ensures alignment with higher timeframe trend
# Volume spike confirms institutional participation, reducing false breakouts
# Discrete position sizing (0.25) minimizes fee churn while maintaining meaningful exposure
# Target: 75-150 total trades over 4 years (19-37/year) on 4h timeframe

name = "4h_Donchian20_1dEMA50_VolumeSpike_v1"
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
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 20-period average volume for spike confirmation (on 4h timeframe)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Donchian channels (20-period) on 4h timeframe
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # 1d EMA50, Donchian warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(highest_20[i]) or np.isnan(lowest_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema_1d = ema_50_1d_aligned[i]
        curr_highest = highest_20[i]
        curr_lowest = lowest_20[i]
        curr_vol_ma = vol_ma_20[i]
        curr_volume = volume[i]
        
        # Volume spike confirmation: current volume > 1.5x 20-period average
        vol_spike = curr_volume > 1.5 * curr_vol_ma
        
        # Donchian breakout conditions
        breakout_long = curr_high > curr_highest   # price breaks above upper band
        breakout_short = curr_low < curr_lowest    # price breaks below lower band
        
        # Handle exits
        if position == 1:  # Long position
            # Exit: price retracement to midpoint OR trend turns bearish
            midpoint = (curr_highest + curr_lowest) / 2
            if (curr_close < midpoint or curr_close < curr_ema_1d):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price retracement to midpoint OR trend turns bullish
            midpoint = (curr_highest + curr_lowest) / 2
            if (curr_close > midpoint or curr_close > curr_ema_1d):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: bullish breakout above upper band AND above 1d EMA50 AND volume spike
            if (breakout_long and 
                curr_close > curr_ema_1d and 
                vol_spike):
                signals[i] = 0.25
                position = 1
            # Short entry: bearish breakout below lower band AND below 1d EMA50 AND volume spike
            elif (breakout_short and 
                  curr_close < curr_ema_1d and 
                  vol_spike):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals