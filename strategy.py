#!/usr/bin/env python3
"""
4h_donchian_breakout_volume_v1
Hypothesis: 4h Donchian breakout with volume confirmation and 1d trend filter.
Works in bull markets via breakouts, in bear markets via short breakdowns.
Volume confirms institutional participation, reducing false breakouts.
Trend filter ensures we trade with higher timeframe momentum.
Target: 20-40 trades/year to avoid overtrading.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4h Donchian channels (20-period)
    lookback = 20
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(lookback, n):
        highest_high[i] = np.max(high[i-lookback:i])
        lowest_low[i] = np.min(low[i-lookback:i])
    
    # Volume confirmation: 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    # 1d trend filter (using close vs 50 EMA)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50 = np.full(len(close_1d), np.nan)
    for i in range(50, len(close_1d)):
        ema_50[i] = np.mean(close_1d[i-50:i])  # Simple MA for efficiency
    
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(ema_50_aligned[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        trend_up = close[i] > ema_50_aligned[i]
        trend_down = close[i] < ema_50_aligned[i]
        
        if position == 1:  # Long
            # Exit: price breaks below Donchian low or trend turns down
            if close[i] < lowest_low[i] or not trend_up:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: price breaks above Donchian high or trend turns up
            if close[i] > highest_high[i] or not trend_down:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long: Donchian breakout with volume and uptrend
            if (close[i] > highest_high[i] and 
                vol_ratio > 1.5 and 
                trend_up):
                position = 1
                signals[i] = 0.25
            # Short: Donchian breakdown with volume and downtrend
            elif (close[i] < lowest_low[i] and 
                  vol_ratio > 1.5 and 
                  trend_down):
                position = -1
                signals[i] = -0.25
    
    return signals