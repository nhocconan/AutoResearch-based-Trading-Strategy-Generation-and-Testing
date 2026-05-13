#!/usr/bin/env python3
name = "6h_MonthlyPivot_Breakout_EMA34"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load monthly data ONCE for pivot calculation
    df_monthly = get_htf_data(prices, '1M')
    if len(df_monthly) < 20:
        return np.zeros(n)
    
    high_m = df_monthly['high'].values
    low_m = df_monthly['low'].values
    close_m = df_monthly['close'].values
    
    # Calculate monthly pivot points
    pivot_m = (high_m + low_m + close_m) / 3.0
    r1_m = 2 * pivot_m - low_m
    s1_m = 2 * pivot_m - high_m
    r2_m = pivot_m + (high_m - low_m)
    s2_m = pivot_m - (high_m - low_m)
    
    # Align monthly pivot levels to 6H timeframe
    pivot_m_aligned = align_htf_to_ltf(prices, df_monthly, pivot_m)
    r1_m_aligned = align_htf_to_ltf(prices, df_monthly, r1_m)
    s1_m_aligned = align_htf_to_ltf(prices, df_monthly, s1_m)
    r2_m_aligned = align_htf_to_ltf(prices, df_monthly, r2_m)
    s2_m_aligned = align_htf_to_ltf(prices, df_monthly, s2_m)
    
    # Load weekly data for trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 30:
        return np.zeros(n)
    
    close_w = df_weekly['close'].values
    ema34_w = pd.Series(close_w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_w_aligned = align_htf_to_ltf(prices, df_weekly, ema34_w)
    
    # 6H EMA34 for trend filter
    close_s = pd.Series(close)
    ema34_6h = close_s.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Volume filter: current volume > 20-period average
    volume_s = pd.Series(volume)
    vol_ma20 = volume_s.rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        if (np.isnan(pivot_m_aligned[i]) or np.isnan(r1_m_aligned[i]) or 
            np.isnan(s1_m_aligned[i]) or np.isnan(r2_m_aligned[i]) or 
            np.isnan(s2_m_aligned[i]) or np.isnan(ema34_w_aligned[i]) or 
            np.isnan(ema34_6h[i])):
            signals[i] = 0.0
            continue
        
        price_above_ema34w = close[i] > ema34_w_aligned[i]
        price_below_ema34w = close[i] < ema34_w_aligned[i]
        price_above_ema34h = close[i] > ema34_6h[i]
        price_below_ema34h = close[i] < ema34_6h[i]
        
        if position == 0:
            # LONG: Break above R2 with volume and weekly uptrend
            if (close[i] > r2_m_aligned[i]) and price_above_ema34w and volume_ok[i]:
                signals[i] = 0.30
                position = 1
            # SHORT: Break below S2 with volume and weekly downtrend
            elif (close[i] < s2_m_aligned[i]) and price_below_ema34w and volume_ok[i]:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price falls back below R1 or weekly trend turns down
            if (close[i] < r1_m_aligned[i]) or (not price_above_ema34w):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # EXIT SHORT: Price rises back above S1 or weekly trend turns up
            if (close[i] > s1_m_aligned[i]) or (not price_below_ema34w):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals