#!/usr/bin/env python3
"""
12h_Camarilla_Pivot_1dTrend_Volume_Spike
Hypothesis: 12h timeframe with daily trend filter (price > 1d EMA34) and daily Camarilla pivot levels (R1/S1).
Long when price touches or crosses above S1 in daily uptrend with volume spike (12h volume > 1.8x 20-period average).
Short when price touches or crosses below R1 in daily downtrend with volume spike.
Designed for 15-35 trades/year to avoid fee drag in 12h timeframe.
Uses volume confirmation to avoid false breaks and trend filter for bias.
Works in bull/bear via daily trend filter and mean reversion at key pivot levels.
"""

name = "12h_Camarilla_Pivot_1dTrend_Volume_Spike"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    # Calculate daily EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate daily OHLC for Camarilla levels (using prior daily bar)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Range and Camarilla levels from prior daily bar
    range_1d = high_1d - low_1d
    r1_1d = close_1d + 1.1 * (range_1d / 12)  # R1 = C + 1.1*(H-L)/12
    s1_1d = close_1d - 1.1 * (range_1d / 12)  # S1 = C - 1.1*(H-L)/12
    
    # Align Camarilla levels to 12h
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Get 12h volume for confirmation
    vol_12h = volume
    vol_ma20_12h = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    vol_ratio_12h = np.divide(vol_12h, vol_ma20_12h, out=np.zeros_like(vol_12h), where=vol_ma20_12h!=0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Warmup
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(r1_1d_aligned[i]) or 
            np.isnan(s1_1d_aligned[i]) or
            np.isnan(vol_ratio_12h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine daily trend using close vs EMA34
        close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
        if np.isnan(close_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        trend_up = close_1d_aligned[i] > ema_34_1d_aligned[i]
        trend_down = close_1d_aligned[i] < ema_34_1d_aligned[i]
        
        if position == 0:
            # Long: price touches/crosses above S1 in daily uptrend with volume spike
            if (low[i] <= s1_1d_aligned[i] and 
                close[i] > s1_1d_aligned[i] and  # reversal confirmation
                vol_ratio_12h[i] > 1.8 and 
                trend_up):
                signals[i] = 0.25
                position = 1
            # Short: price touches/crosses below R1 in daily downtrend with volume spike
            elif (high[i] >= r1_1d_aligned[i] and 
                  close[i] < r1_1d_aligned[i] and  # reversal confirmation
                  vol_ratio_12h[i] > 1.8 and 
                  trend_down):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price touches/crosses below S1 or trend turns down
            if (low[i] <= s1_1d_aligned[i] and 
                close[i] > s1_1d_aligned[i]) or \
               not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price touches/crosses above R1 or trend turns up
            if (high[i] >= r1_1d_aligned[i] and 
                close[i] < r1_1d_aligned[i]) or \
               not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals