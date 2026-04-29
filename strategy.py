#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d ATR regime filter and volume spike confirmation
# Donchian channels provide adaptive support/resistance that works in both bull and bear markets
# 1d ATR regime filter identifies high volatility environments where breakouts are more reliable
# Volume spike confirms institutional participation in the breakout
# Target: 50-150 total trades over 4 years (12-37/year) on 12h timeframe

name = "12h_Donchian20_ATRRegime_VolumeSpike"
timeframe = "12h"
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
    
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d ATR(14) for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate ATR percentile rank over 50 days for regime identification
    atr_percentile = pd.Series(atr_14_1d).rolling(window=50, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else np.nan, raw=False
    ).values
    atr_percentile_aligned = align_htf_to_ltf(prices, df_1d, atr_percentile)
    
    # Calculate Donchian channels (20-period) on 12h timeframe
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)  # Donchian and ATR percentile warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_14_1d_aligned[i]) or np.isnan(atr_percentile_aligned[i]) or 
            np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_vol_ma = vol_ma_20[i]
        curr_atr_percentile = atr_percentile_aligned[i]
        curr_highest_20 = highest_20[i]
        curr_lowest_20 = lowest_20[i]
        
        # Volume confirmation: current volume > 1.8x 20-period average
        vol_confirm = curr_volume > 1.8 * curr_vol_ma
        
        # Regime filter: ATR percentile > 0.4 (avoid extremely low volatility environments)
        regime_ok = curr_atr_percentile > 0.4
        
        # Handle exits
        if position == 1:  # Long position
            # Exit: price closes below Donchian lower band (20-period low)
            if curr_close < curr_lowest_20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian upper band (20-period high)
            if curr_close > curr_highest_20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: price breaks above Donchian upper band + volume confirmation + regime filter
            if (curr_close > curr_highest_20 and 
                vol_confirm and 
                regime_ok):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian lower band + volume confirmation + regime filter
            elif (curr_close < curr_lowest_20 and 
                  vol_confirm and 
                  regime_ok):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals