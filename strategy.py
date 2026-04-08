#!/usr/bin/env python3
"""
6h_1d_donchian_breakout_volume_v1
Hypothesis: Use 1d trend via EMA(50), 6h Donchian channel breakout (20-period) with volume confirmation.
Long when price breaks above upper Donchian band in 1d uptrend, short when breaks below lower band in 1d downtrend.
Exit when price crosses EMA(50) or opposite Donchian band is touched.
Designed for medium frequency (~50-100 trades/year) to balance edge and fees.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_donchian_breakout_volume_v1"
timeframe = "6h"
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
    
    # 6h Donchian channel (20-period)
    lookback = 20
    upper = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lower = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume confirmation: volume > 1.5x average of last 12 periods (12*6h = 3 days)
    vol_ma = pd.Series(volume).rolling(window=12, min_periods=12).mean().values
    vol_confirm = volume > vol_ma * 1.5
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = max(100, lookback)
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(ema_50_aligned[i]) or np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below EMA(50) or touches lower Donchian band
            if close[i] <= ema_50_aligned[i] or close[i] <= lower[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
                
        elif position == -1:  # Short position
            # Exit: price crosses above EMA(50) or touches upper Donchian band
            if close[i] >= ema_50_aligned[i] or close[i] >= upper[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Long entry: price breaks above upper Donchian with volume and 1d uptrend
            if (close[i] > upper[i] and 
                ema_50_aligned[i] > ema_50_aligned[max(0, i-2)] and  # Uptrend confirmation
                vol_confirm[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below lower Donchian with volume and 1d downtrend
            elif (close[i] < lower[i] and 
                  ema_50_aligned[i] < ema_50_aligned[max(0, i-2)] and  # Downtrend confirmation
                  vol_confirm[i]):
                position = -1
                signals[i] = -0.25
    
    return signals