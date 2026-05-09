#!/usr/bin/env python3
"""
1d_Camarilla_Pivot_R1_S1_Breakout_1wTrend_Volume
Camarilla pivot R1/S1 breakout on daily chart with weekly trend filter and volume confirmation.
Breakouts above R1 in weekly uptrend or below S1 in weekly downtrend.
Designed for low trade frequency (<20/year) to minimize fee drag in 2025 bear market.
Works in bull (breakouts continue) and bear (mean reversion at extremes) via trend filter.
"""

name = "1d_Camarilla_Pivot_R1_S1_Breakout_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Calculate weekly 5-period EMA for trend filter
    close_1w = df_1w['close'].values
    ema_5_1w = pd.Series(close_1w).ewm(span=5, adjust=False, min_periods=5).mean().values
    ema_5_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_5_1w)
    
    # Calculate Camarilla pivot levels on daily data
    # Pivot = (H + L + C) / 3
    # R1 = C + (H - L) * 1.1 / 12
    # S1 = C - (H - L) * 1.1 / 12
    pivot = (high + low + close) / 3.0
    r1 = close + (high - low) * 1.1 / 12.0
    s1 = close - (high - low) * 1.1 / 12.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 1  # Need at least 1 day for pivot calculation
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if np.isnan(ema_5_1w_aligned[i]) or np.isnan(pivot[i]) or np.isnan(r1[i]) or np.isnan(s1[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_1w = ema_5_1w_aligned[i]
        piv = pivot[i]
        r1_val = r1[i]
        s1_val = s1[i]
        vol = volume[i]
        
        # Calculate 5-day volume average for spike detection
        if i >= 5:
            vol_ma = np.mean(volume[i-5:i])
        else:
            vol_ma = np.mean(volume[:i]) if i > 0 else volume[i]
        
        if position == 0:
            # Enter long: Close > R1 AND price > weekly EMA5 (uptrend) AND volume > 2x average
            if close[i] > r1_val and close[i] > ema_1w and vol > 2.0 * vol_ma:
                signals[i] = 0.25
                position = 1
            # Enter short: Close < S1 AND price < weekly EMA5 (downtrend) AND volume > 2x average
            elif close[i] < s1_val and close[i] < ema_1w and vol > 2.0 * vol_ma:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Close < pivot OR trend reverses (price < weekly EMA5)
            if close[i] < piv or close[i] < ema_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Close > pivot OR trend reverses (price > weekly EMA5)
            if close[i] > piv or close[i] > ema_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals