#!/usr/bin/env python3
"""
4h_Donchian_Breakout_1d_Trend_Volume_v2
Hypothesis: Donchian(20) breakout on 4h with 1d EMA(50) trend filter and volume confirmation.
Works in bull via upward breakouts in uptrend, works in bear via downward breakouts in downtrend.
Volume filters out false breakouts. Target: 20-40 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Donchian_Breakout_1d_Trend_Volume_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # 1d EMA(50) for trend
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Donchian(20) on 4h: highest high/lowest low of last 20 periods
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume confirmation: volume > 1.5x average of last 20 periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > vol_ma * 1.5
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(ema_50_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price returns to Donchian middle or trend changes
            donchian_mid = (highest_high[i] + lowest_low[i]) / 2
            if close[i] <= donchian_mid or ema_50_aligned[i] < ema_50_aligned[max(0, i-1)]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
                
        elif position == -1:  # Short position
            # Exit: price returns to Donchian middle or trend changes
            donchian_mid = (highest_high[i] + lowest_low[i]) / 2
            if close[i] >= donchian_mid or ema_50_aligned[i] > ema_50_aligned[max(0, i-1)]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Long entry: price breaks above Donchian upper with volume and 1d uptrend
            if (not np.isnan(highest_high[i]) and close[i] > highest_high[i] and 
                ema_50_aligned[i] > ema_50_aligned[max(0, i-1)] and  # Uptrend confirmation
                vol_confirm[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below Donchian lower with volume and 1d downtrend
            elif (not np.isnan(lowest_low[i]) and close[i] < lowest_low[i] and 
                  ema_50_aligned[i] < ema_50_aligned[max(0, i-1)] and  # Downtrend confirmation
                  vol_confirm[i]):
                position = -1
                signals[i] = -0.25
    
    return signals