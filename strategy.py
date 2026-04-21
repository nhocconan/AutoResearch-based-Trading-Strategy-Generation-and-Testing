#!/usr/bin/env python3
"""
1d_1W_Weekly_Camarilla_R1S1_Breakout_Volume_TrendFilter_v1
Hypothesis: Weekly timeframe determines trend direction via price position relative to 20-week SMA.
Daily timeframe provides entry signals via breakouts of weekly-derived Camarilla levels (R1, S1) with volume confirmation.
In uptrend (price > weekly SMA20), only long breakouts above R1 are taken.
In downtrend (price < weekly SMA20), only short breakdowns below S1 are taken.
This avoids counter-trend trades in strong trends, reducing whipsaws. Volume filter ensures breakout strength.
Designed for 1d timeframe to capture multi-week moves with ~10-25 trades/year.
Works in bull markets by buying breakouts and in bear markets by selling breakdowns.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data once for trend filter and Camarilla pivots
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly SMA20 for trend filter
    close_s_1w = pd.Series(close_1w)
    sma20_1w = close_s_1w.rolling(window=20, min_periods=20).mean().values
    
    # Weekly Camarilla pivot levels (based on previous week)
    pp = np.full_like(close_1w, np.nan)
    r1 = np.full_like(close_1w, np.nan)
    s1 = np.full_like(close_1w, np.nan)
    
    for i in range(1, len(high_1w)):
        pp[i] = (high_1w[i-1] + low_1w[i-1] + close_1w[i-1]) / 3.0
        r1[i] = close_1w[i-1] + (high_1w[i-1] - low_1w[i-1]) * 1.1 / 12.0
        s1[i] = close_1w[i-1] - (high_1w[i-1] - low_1w[i-1]) * 1.1 / 12.0
    
    # Shift to align with current week (levels based on previous week)
    pp = np.roll(pp, 1)
    r1 = np.roll(r1, 1)
    s1 = np.roll(s1, 1)
    pp[0] = np.nan
    r1[0] = np.nan
    s1[0] = np.nan
    
    # Align weekly indicators to daily timeframe
    sma20_1w_aligned = align_htf_to_ltf(prices, df_1w, sma20_1w)
    pp_aligned = align_htf_to_ltf(prices, df_1w, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(sma20_1w_aligned[i]) or np.isnan(pp_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume filter: current volume > 1.5 * 20-day average
        if i >= 20:
            vol_ma = prices['volume'].iloc[i-20:i].mean()
            volume_ok = volume > 1.5 * vol_ma
        else:
            volume_ok = False
        
        # Trend filter: price relative to weekly SMA20
        uptrend = price > sma20_1w_aligned[i]
        downtrend = price < sma20_1w_aligned[i]
        
        if position == 0:
            # Long conditions: break above R1 + volume confirmation + uptrend filter
            if price > r1_aligned[i] and volume_ok and uptrend:
                signals[i] = 0.25
                position = 1
            # Short conditions: break below S1 + volume confirmation + downtrend filter
            elif price < s1_aligned[i] and volume_ok and downtrend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses back below weekly pivot point
            if price < pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses back above weekly pivot point
            if price > pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_1W_Weekly_Camarilla_R1S1_Breakout_Volume_TrendFilter_v1"
timeframe = "1d"
leverage = 1.0