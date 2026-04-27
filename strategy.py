#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_Volume
Breakout of Camarilla R1/S1 with 1-day trend filter and volume confirmation.
In uptrend (price > 1-day EMA34): buy breakouts above R1.
In downtrend (price < 1-day EMA34): sell breakdowns below S1.
Volume filter ensures participation. Target: 12-37 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels"""
    pivot = (high + low + close) / 3
    range_val = high - low
    r1 = close + range_val * 1.1 / 12
    s1 = close - range_val * 1.1 / 12
    r2 = close + range_val * 1.1 / 6
    s2 = close - range_val * 1.1 / 6
    r3 = close + range_val * 1.1 / 4
    s3 = close - range_val * 1.1 / 4
    return pivot, r1, s1, r2, s2, r3, s3

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for trend and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Daily EMA34 for trend
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Daily volume average for confirmation
    vol_1d = df_1d['volume'].values
    vol_avg_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    
    # Get daily data for Camarilla levels
    df_1d_cam = get_htf_data(prices, '1d')
    if len(df_1d_cam) < 2:
        return np.zeros(n)
    
    # Calculate daily Camarilla levels
    high_1d = df_1d_cam['high'].values
    low_1d = df_1d_cam['low'].values
    close_1d = df_1d_cam['close'].values
    
    pivot, r1, s1, r2, s2, r3, s3 = calculate_camarilla(high_1d, low_1d, close_1d)
    
    # Align Camarilla levels
    pivot_aligned = align_htf_to_ltf(prices, df_1d_cam, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d_cam, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d_cam, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d_cam, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d_cam, s2)
    r3_aligned = align_htf_to_ltf(prices, df_1d_cam, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d_cam, s3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need EMA34 (34) + volume avg (20)
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_avg_20_1d_aligned[i]) or
            np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current 12h price and volume
        price_now = close[i]
        vol_now = volume[i]
        vol_avg = vol_avg_20_1d_aligned[i]
        
        # Current levels
        ema_trend = ema_34_1d_aligned[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        
        # Volume filter: volume > 1.1x daily average
        vol_filter = vol_now > 1.1 * vol_avg
        
        if position == 0:
            # Uptrend: buy breakout above R1
            if price_now > ema_trend and price_now > r1_val and vol_filter:
                signals[i] = size
                position = 1
            # Downtrend: sell breakdown below S1
            elif price_now < ema_trend and price_now < s1_val and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: trend reverses or price returns to pivot
            if price_now < ema_trend or price_now < pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: trend reverses or price returns to pivot
            if price_now > ema_trend or price_now > pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0