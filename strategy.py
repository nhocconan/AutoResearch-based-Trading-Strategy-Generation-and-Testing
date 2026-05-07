#!/usr/bin/env python3
"""
1H_EMA_Crossover_4hTrend_1dVolFilter_v1
Hypothesis: Use 1h EMA crossover for entry timing, filtered by 4h EMA trend direction and 1d volume spike.
Long when 1h EMA(8) crosses above EMA(21) AND 4h EMA(50) is rising AND 1d volume > 1.5x 20-day average.
Short when 1h EMA(8) crosses below EMA(21) AND 4h EMA(50) is falling AND 1d volume > 1.5x 20-day average.
This combines faster entry timing with higher timeframe trend and volume confirmation to reduce false signals and work in both bull and bear markets.
"""
name = "1H_EMA_Crossover_4hTrend_1dVolFilter_v1"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1h EMA(8) and EMA(21)
    close_s = pd.Series(close)
    ema8 = close_s.ewm(span=8, adjust=False, min_periods=8).values
    ema21 = close_s.ewm(span=21, adjust=False, min_periods=21).values
    
    # Get 4h data for EMA(50) trend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 10:
        return np.zeros(n)
    
    close_4h = pd.Series(df_4h['close'])
    ema50_4h = close_4h.ewm(span=50, adjust=False, min_periods=50).values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Get 1d data for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    vol_1d = pd.Series(df_1d['volume'])
    vol_avg_1d = vol_1d.rolling(window=20, min_periods=20).mean().values
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_exit = 0  # bars since last exit to prevent overtrading
    
    start_idx = max(21, 50)  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        bars_since_exit += 1
        
        # Skip if any data is not ready
        if (np.isnan(ema8[i]) or np.isnan(ema21[i]) or np.isnan(ema50_4h_aligned[i]) or 
            np.isnan(vol_avg_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            continue
        
        if position == 0:
            # Minimum 6 bars between trades to reduce frequency
            if bars_since_exit < 6:
                continue
                
            # Check 1d volume filter
            vol_filter = volume[i] > (vol_avg_1d_aligned[i] * 1.5)
            
            # Long: EMA8 crosses above EMA21 AND 4h EMA50 rising AND volume filter
            if (ema8[i] > ema21[i] and ema8[i-1] <= ema21[i-1] and 
                ema50_4h_aligned[i] > ema50_4h_aligned[i-1] and vol_filter):
                signals[i] = 0.20
                position = 1
                bars_since_exit = 0
            # Short: EMA8 crosses below EMA21 AND 4h EMA50 falling AND volume filter
            elif (ema8[i] < ema21[i] and ema8[i-1] >= ema21[i-1] and 
                  ema50_4h_aligned[i] < ema50_4h_aligned[i-1] and vol_filter):
                signals[i] = -0.20
                position = -1
                bars_since_exit = 0
        elif position != 0:
            # Exit: EMA8 crosses back in opposite direction
            if position == 1 and ema8[i] < ema21[i]:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            elif position == -1 and ema8[i] > ema21[i]:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            else:
                # Hold position
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals