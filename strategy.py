#!/usr/bin/env python3
"""
1d_1w_ema_crossover_volume_v1
Hypothesis: Use 1w EMA crossover for long-term trend direction, 1d EMA crossover for entry timing, and volume confirmation. 
Only trade when 1d EMA(8) crosses above/below EMA(21) in the direction of 1w trend (EMA(50)). 
Exit when 1d EMA(8) crosses back or 1w trend changes.
Designed for fewer trades (<25/year) to avoid fee decay and work in both bull/bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_ema_crossover_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for long-term trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Get 1d data for entry signals
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA(8) and EMA(21) for entry
    close_1d = df_1d['close'].values
    ema_8_1d = pd.Series(close_1d).ewm(span=8, adjust=False, min_periods=8).mean().values
    ema_21_1d = pd.Series(close_1d).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Align 1d EMAs to 1d timeframe (no shift needed as they are on same timeframe)
    ema_8_1d_aligned = ema_8_1d  # Already on 1d timeframe
    ema_21_1d_aligned = ema_21_1d  # Already on 1d timeframe
    
    # Volume confirmation: volume > 1.5x average of last 20 days
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > vol_ma * 1.5
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(ema_8_1d_aligned[i]) or 
            np.isnan(ema_21_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: EMA(8) crosses below EMA(21) or 1w trend turns down
            if ema_8_1d_aligned[i] < ema_21_1d_aligned[i] or ema_50_1w_aligned[i] < close[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
                
        elif position == -1:  # Short position
            # Exit: EMA(8) crosses above EMA(21) or 1w trend turns up
            if ema_8_1d_aligned[i] > ema_21_1d_aligned[i] or ema_50_1w_aligned[i] > close[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Long entry: EMA(8) crosses above EMA(21) with volume and 1w uptrend
            if (ema_8_1d_aligned[i] > ema_21_1d_aligned[i] and 
                ema_8_1d_aligned[i-1] <= ema_21_1d_aligned[i-1] and  # Cross just happened
                ema_50_1w_aligned[i] < close[i] and  # 1w uptrend (price above EMA50)
                vol_confirm[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: EMA(8) crosses below EMA(21) with volume and 1w downtrend
            elif (ema_8_1d_aligned[i] < ema_21_1d_aligned[i] and 
                  ema_8_1d_aligned[i-1] >= ema_21_1d_aligned[i-1] and  # Cross just happened
                  ema_50_1w_aligned[i] > close[i] and  # 1w downtrend (price below EMA50)
                  vol_confirm[i]):
                position = -1
                signals[i] = -0.25
    
    return signals