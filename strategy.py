#!/usr/bin/env python3
"""
Hypothesis: On 12h timeframe, weekly pivot levels act as strong support/resistance zones. 
Price often reverses or accelerates when touching weekly R1/S1 levels with volume confirmation. 
The strategy enters long when price crosses above weekly R1 with volume > 2.0x average and 
price above weekly EMA50 trend filter, and short when price crosses below weekly S1 with 
volume > 2.0x average and price below weekly EMA50. Exits occur at the weekly pivot level. 
Designed for 12h timeframe to capture multi-day moves in both bull (breakouts) and bear 
(reversals at resistance) regimes with ~15-25 trades per year.
"""

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
    
    # Get weekly data for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot and support/resistance levels
    whigh = df_1w['high'].values
    wlow = df_1w['low'].values
    wclose = df_1w['close'].values
    
    pivot = (whigh + wlow + wclose) / 3
    range_ = whigh - wlow
    r1 = 2 * pivot - wlow
    s1 = 2 * pivot - whigh
    
    # Calculate weekly EMA50 for trend filter
    ema_50 = pd.Series(wclose).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all weekly levels to 12h timeframe (waits for weekly bar to close)
    pivot_12h = align_htf_to_ltf(prices, df_1w, pivot)
    r1_12h = align_htf_to_ltf(prices, df_1w, r1)
    s1_12h = align_htf_to_ltf(prices, df_1w, s1)
    ema_50_12h = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Volume confirmation: 30-period volume MA on 12h
    volume_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean()
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 60  # warmup for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(pivot_12h[i]) or np.isnan(r1_12h[i]) or np.isnan(s1_12h[i]) or
            np.isnan(ema_50_12h[i]) or np.isnan(volume_ma_30.iloc[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = volume_ma_30.iloc[i]
        
        if position == 0:
            # Long: price crosses above R1 with volume spike and above weekly EMA50
            if price > r1_12h[i] and vol > 2.0 * vol_ma and price > ema_50_12h[i]:
                signals[i] = 0.25
                position = 1
            # Short: price crosses below S1 with volume spike and below weekly EMA50
            elif price < s1_12h[i] and vol > 2.0 * vol_ma and price < ema_50_12h[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to weekly pivot level
            if price < pivot_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to weekly pivot level
            if price > pivot_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WeeklyPivot_R1S1_Volume_EMA50"
timeframe = "12h"
leverage = 1.0