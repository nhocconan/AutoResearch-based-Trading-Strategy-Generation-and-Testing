#!/usr/bin/env python3
# 6h_12h_donchian_breakout_volume_v1
# Hypothesis: Trade 6h Donchian channel breakouts with 12h trend filter and volume confirmation.
# Long when price breaks above 6h Donchian upper band (20-period) with volume > 1.5x 20-period average and 12h close > 12h EMA50 (uptrend).
# Short when price breaks below 6h Donchian lower band with volume surge and 12h close < 12h EMA50 (downtrend).
# Exit when price returns to 6h Donchian middle band (20-period EMA of close).
# Designed for 6h timeframe to target 12-37 trades/year (50-150 total over 4 years).
# Uses 12h trend filter to avoid counter-trend trades, working in both bull and bear markets by aligning with higher timeframe momentum.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_12h_donchian_breakout_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # EMA50 for 12h trend
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # 6h Donchian channels (20-period)
    high_6h = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_6h = pd.Series(low).rolling(window=20, min_periods=20).min().values
    # Middle band: 20-period EMA of close
    middle_6h = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 50  # Ensure EMA50_12h is ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema50_12h_aligned[i]) or np.isnan(high_6h[i]) or 
            np.isnan(low_6h[i]) or np.isnan(middle_6h[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Volume surge condition
        vol_surge = volume[i] > 1.5 * vol_ma_20[i] if vol_ma_20[i] > 0 else False
        
        if position == 1:  # Long position
            # Exit: price returns to middle band
            if close[i] <= middle_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price returns to middle band
            if close[i] >= middle_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above upper band with volume surge and 12h uptrend
            if (close[i] > high_6h[i] and vol_surge and 
                close[i] > ema50_12h_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below lower band with volume surge and 12h downtrend
            elif (close[i] < low_6h[i] and vol_surge and 
                  close[i] < ema50_12h_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals