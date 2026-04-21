#!/usr/bin/env python3
"""
1d_1w_Camarilla_R1S1_Breakout_Volume
Hypothesis: Use weekly market structure (higher timeframe) to filter daily breakouts.
Long when price breaks above daily R1 with volume confirmation AND weekly trend is up (price > weekly EMA20).
Short when price breaks below daily S1 with volume confirmation AND weekly trend is down (price < weekly EMA20).
Exit when price crosses back through the daily pivot point.
Weekly trend filter reduces false breakouts in choppy markets, improving win rate.
Designed for 1d timeframe to capture multi-day moves with ~10-25 trades/year.
Works in bull markets by buying breakouts and in bear markets by selling breakdowns.
Volume confirmation filters false breakouts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data once for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Load weekly data once for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Daily OHLC for Camarilla calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Weekly close for EMA calculation
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA20 for trend filter
    close_1w_series = pd.Series(close_1w)
    ema_20_1w = close_1w_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Daily Camarilla pivot levels (based on previous day)
    pp = np.full_like(close_1d, np.nan)
    r1 = np.full_like(close_1d, np.nan)
    s1 = np.full_like(close_1d, np.nan)
    
    for i in range(1, len(high_1d)):
        pp[i] = (high_1d[i-1] + low_1d[i-1] + close_1d[i-1]) / 3.0
        r1[i] = close_1d[i-1] + (high_1d[i-1] - low_1d[i-1]) * 1.1 / 12.0
        s1[i] = close_1d[i-1] - (high_1d[i-1] - low_1d[i-1]) * 1.1 / 12.0
    
    # Shift to align with current day (levels are based on previous day)
    pp = np.roll(pp, 1)
    r1 = np.roll(r1, 1)
    s1 = np.roll(s1, 1)
    pp[0] = np.nan
    r1[0] = np.nan
    s1[0] = np.nan
    
    # Align all indicators to lower timeframe (1d)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_20_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume filter: current volume > 1.5 * 20-period average
        if i >= 20:
            vol_ma = prices['volume'].iloc[i-20:i].mean()
            volume_ok = volume > 1.5 * vol_ma
        else:
            volume_ok = False
        
        # Weekly trend filter
        weekly_uptrend = price > ema_20_1w_aligned[i]
        weekly_downtrend = price < ema_20_1w_aligned[i]
        
        if position == 0:
            # Long conditions: break above R1 + volume confirmation + weekly uptrend
            if price > r1_aligned[i] and volume_ok and weekly_uptrend:
                signals[i] = 0.25
                position = 1
            # Short conditions: break below S1 + volume confirmation + weekly downtrend
            elif price < s1_aligned[i] and volume_ok and weekly_downtrend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses back below pivot point
            if price < pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses back above pivot point
            if price > pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_1w_Camarilla_R1S1_Breakout_Volume"
timeframe = "1d"
leverage = 1.0