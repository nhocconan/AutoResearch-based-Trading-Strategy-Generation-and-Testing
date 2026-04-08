#!/usr/bin/env python3
# 6h_1w_1d_cci_breakout_v1
# Hypothesis: Use weekly CCI to identify trend direction and daily CCI pullbacks for entries.
# In weekly uptrend (CCI > 0): look for long entries when daily CCI pulls back to oversold (< -100) with volume.
# In weekly downtrend (CCI < 0): look for short entries when daily CCI pulls back to overbought (> 100) with volume.
# Exit when CCI reverses or trend changes. Uses 60-period CCI for stability and volume confirmation to avoid false signals.
# Target: 20-50 trades/year (80-200 total over 4 years) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_1d_cci_breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly CCI for trend filter (60-period)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate CCI: (Typical Price - SMA) / (0.015 * Mean Deviation)
    tp_1w = (high_1w + low_1w + close_1w) / 3
    sma_1w = pd.Series(tp_1w).rolling(window=60, min_periods=60).mean().values
    mad_1w = pd.Series(tp_1w).rolling(window=60, min_periods=60).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True).values
    cci_1w = (tp_1w - sma_1w) / (0.015 * mad_1w)
    cci_1w[np.isnan(mad_1w) | (mad_1w == 0)] = 0
    cci_1w_aligned = align_htf_to_ltf(prices, df_1w, cci_1w)
    
    # Daily CCI for entry signals (20-period)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tp_1d = (high_1d + low_1d + close_1d) / 3
    sma_1d = pd.Series(tp_1d).rolling(window=20, min_periods=20).mean().values
    mad_1d = pd.Series(tp_1d).rolling(window=20, min_periods=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True).values
    cci_1d = (tp_1d - sma_1d) / (0.015 * mad_1d)
    cci_1d[np.isnan(mad_1d) | (mad_1d == 0)] = 0
    cci_1d_aligned = align_htf_to_ltf(prices, df_1d, cci_1d)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 100  # Ensure all indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(cci_1w_aligned[i]) or np.isnan(cci_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Volume surge condition
        vol_surge = volume[i] > 1.5 * vol_ma_20[i] if vol_ma_20[i] > 0 else False
        
        if position == 1:  # Long position
            # Exit: CCI turns overbought (> 100) or weekly trend turns down (CCI < 0)
            if cci_1d_aligned[i] > 100 or cci_1w_aligned[i] < 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: CCI turns oversold (< -100) or weekly trend turns up (CCI > 0)
            if cci_1d_aligned[i] < -100 or cci_1w_aligned[i] > 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: weekly uptrend + daily CCI oversold pullback + volume
            if (cci_1w_aligned[i] > 0 and  # Weekly uptrend
                cci_1d_aligned[i] < -100 and  # Daily oversold
                vol_surge):
                position = 1
                signals[i] = 0.25
            # Short entry: weekly downtrend + daily CCI overbought pullback + volume
            elif (cci_1w_aligned[i] < 0 and  # Weekly downtrend
                  cci_1d_aligned[i] > 100 and  # Daily overbought
                  vol_surge):
                position = -1
                signals[i] = -0.25
    
    return signals