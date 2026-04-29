#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume spike (>2x 20-period average)
# Donchian channels capture significant price breakouts with institutional follow-through.
# 1w EMA50 ensures alignment with higher timeframe trend to avoid counter-trend whipsaws in both bull/bear markets.
# Volume spike filter (>2x average) confirms significant market interest, reducing false breakouts.
# Discrete position sizing (0.25) minimizes fee churn while maintaining meaningful exposure.
# Target: 30-100 total trades over 4 years (7-25/year) on 1d timeframe.

name = "1d_Donchian20_1wEMA50_VolumeSpike_v1"
timeframe = "1d"
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
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Donchian channels (20-period) on 1d timeframe
    # Upper band: 20-period high, Lower band: 20-period low
    high_ma_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_ma_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 20-period average volume for spike confirmation (on 1d timeframe)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # 1w EMA50, Donchian/volume MA warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(high_ma_20[i]) or 
            np.isnan(low_ma_20[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema_1w = ema_50_1w_aligned[i]
        curr_upper = high_ma_20[i]
        curr_lower = low_ma_20[i]
        curr_vol_ma = vol_ma_20[i]
        curr_volume = volume[i]
        
        # Volume spike confirmation: current volume > 2x 20-period average
        vol_spike = curr_volume > 2.0 * curr_vol_ma
        
        # Donchian breakout conditions
        breakout_long = curr_high > curr_upper   # price breaks above upper band
        breakout_short = curr_low < curr_lower   # price breaks below lower band
        
        # Handle exits
        if position == 1:  # Long position
            # Exit: price retracement to midpoint OR trend turns bearish
            midpoint = (curr_upper + curr_lower) / 2.0
            if (not np.isnan(midpoint) and 
                (curr_close < midpoint or curr_close < curr_ema_1w)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price retracement to midpoint OR trend turns bullish
            midpoint = (curr_upper + curr_lower) / 2.0
            if (not np.isnan(midpoint) and 
                (curr_close > midpoint or curr_close > curr_ema_1w)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: bullish breakout above upper band AND above 1w EMA50 AND volume spike
            if (breakout_long and 
                curr_close > curr_ema_1w and 
                vol_spike):
                signals[i] = 0.25
                position = 1
            # Short entry: bearish breakout below lower band AND below 1w EMA50 AND volume spike
            elif (breakout_short and 
                  curr_close < curr_ema_1w and 
                  vol_spike):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals