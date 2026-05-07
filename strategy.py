#!/usr/bin/env python3
"""
1d_Weekly_Camarilla_Pivot_With_Volume_Spike
Hypothesis: Trade long when price touches Camarilla S1 on a weekly bullish trend (price > weekly EMA34) with volume spike (>2x average); trade short when price touches R1 on a weekly bearish trend (price < weekly EMA34) with volume spike. Uses weekly timeframe for trend direction and daily for precise entry. Designed to work in both bull and bear markets by requiring trend alignment and volume confirmation. Targets 15-30 trades/year with low fee impact.
"""

name = "1d_Weekly_Camarilla_Pivot_With_Volume_Spike"
timeframe = "1d"
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
    
    # Get weekly data for trend filter and Camarilla calculation
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 34:
        return np.zeros(n)
    
    weekly_high = df_w['high'].values
    weekly_low = df_w['low'].values
    weekly_close = df_w['close'].values
    
    # Calculate weekly EMA34 for trend filter
    ema_34_w = pd.Series(weekly_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_w_aligned = align_htf_to_ltf(prices, df_w, ema_34_w)
    
    # Calculate Camarilla levels from previous weekly bar
    # Camarilla: H4 = C + 1.1*(H-L)/2, L4 = C - 1.1*(H-L)/2, etc.
    # We'll use S1 and R1 which are closer to price
    # S1 = C - 1.1*(H-L)/6, R1 = C + 1.1*(H-L)/6
    hl_range = weekly_high - weekly_low
    s1 = weekly_close - 1.1 * hl_range / 6
    r1 = weekly_close + 1.1 * hl_range / 6
    
    # Align Camarilla levels to daily timeframe (use previous weekly bar's levels)
    s1_aligned = align_htf_to_ltf(prices, df_w, s1)
    r1_aligned = align_htf_to_ltf(prices, df_w, r1)
    
    # Get daily volume for confirmation
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.divide(volume, vol_ma20, out=np.zeros_like(volume), where=vol_ma20!=0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(s1_aligned[i]) or 
            np.isnan(r1_aligned[i]) or 
            np.isnan(ema_34_w_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine weekly trend
        trend_up = weekly_close[i] > ema_34_w_aligned[i]  # Note: using current weekly close for trend
        trend_down = weekly_close[i] < ema_34_w_aligned[i]
        
        if position == 0:
            # Long: price touches or goes below S1 with weekly uptrend and volume spike
            if (low[i] <= s1_aligned[i] and 
                trend_up and 
                vol_ratio[i] > 2.0):
                signals[i] = 0.25
                position = 1
            # Short: price touches or goes above R1 with weekly downtrend and volume spike
            elif (high[i] >= r1_aligned[i] and 
                  trend_down and 
                  vol_ratio[i] > 2.0):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to weekly EMA34 or trend turns down
            if (close[i] >= ema_34_w_aligned[i] or not trend_up):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to weekly EMA34 or trend turns up
            if (close[i] <= ema_34_w_aligned[i] or not trend_down):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals