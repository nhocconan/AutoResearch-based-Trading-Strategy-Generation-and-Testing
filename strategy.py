# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# 4h_Donchian_Breakout_Volume_Trend_Filter
# Hypothesis: Breakout of 20-period Donchian channel with volume confirmation and trend filter.
# In bull markets (price > 1d EMA50): long on upper band breakout with volume spike.
# In bear markets (price < 1d EMA50): short on lower band breakout with volume spike.
# Volume confirmation: current volume > 1.5x 20-period average volume.
# Trend filter: EMA50 on 1d timeframe to avoid counter-trend trades.
# Target: 20-50 trades/year to minimize fee drag.

name = "4h_Donchian_Breakout_Volume_Trend_Filter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtrader.mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate Donchian channels (20-period)
    period = 20
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    # Calculate average volume (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(period - 1, n):
        vol_ma[i] = np.mean(volume[i - period + 1:i + 1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(period, 50)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(ema50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_ok = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Determine trend from 1d EMA50
            uptrend = close[i] > ema50_1d_aligned[i]
            downtrend = close[i] < ema50_1d_aligned[i]
            
            # Long: uptrend + price breaks above upper band + volume confirmation
            if uptrend and close[i] > upper[i] and volume_ok:
                signals[i] = 0.25
                position = 1
            # Short: downtrend + price breaks below lower band + volume confirmation
            elif downtrend and close[i] < lower[i] and volume_ok:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below lower band or trend reverses
            if close[i] < lower[i] or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above upper band or trend reverses
            if close[i] > upper[i] or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals