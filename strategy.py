#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data once for pivot levels and ATR
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 15:
        return np.zeros(n)
    
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily = df_daily['close'].values
    volume_daily = df_daily['volume'].values
    
    # Daily ATR (14)
    tr1 = np.abs(high_daily - low_daily)
    tr2 = np.abs(high_daily - np.roll(close_daily, 1))
    tr3 = np.abs(low_daily - np.roll(close_daily, 1))
    tr1[0] = high_daily[0] - low_daily[0]
    tr2[0] = np.abs(high_daily[0] - close_daily[0])
    tr3[0] = np.abs(low_daily[0] - close_daily[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_daily = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_daily_aligned = align_htf_to_ltf(prices, df_daily, atr_daily)
    
    # Daily pivot levels (standard)
    pivot_daily = (high_daily + low_daily + close_daily) / 3.0
    r1_daily = 2 * pivot_daily - low_daily
    s1_daily = 2 * pivot_daily - high_daily
    r2_daily = pivot_daily + (high_daily - low_daily)
    s2_daily = pivot_daily - (high_daily - low_daily)
    r3_daily = high_daily + 2 * (pivot_daily - low_daily)
    s3_daily = low_daily - 2 * (high_daily - pivot_daily)
    r4_daily = pivot_daily + 3 * (high_daily - low_daily)
    s4_daily = pivot_daily - 3 * (high_daily - low_daily)
    
    # Align pivot levels
    pivot_daily_aligned = align_htf_to_ltf(prices, df_daily, pivot_daily)
    r1_daily_aligned = align_htf_to_ltf(prices, df_daily, r1_daily)
    s1_daily_aligned = align_htf_to_ltf(prices, df_daily, s1_daily)
    r2_daily_aligned = align_htf_to_ltf(prices, df_daily, r2_daily)
    s2_daily_aligned = align_htf_to_ltf(prices, df_daily, s2_daily)
    r3_daily_aligned = align_htf_to_ltf(prices, df_daily, r3_daily)
    s3_daily_aligned = align_htf_to_ltf(prices, df_daily, s3_daily)
    r4_daily_aligned = align_htf_to_ltf(prices, df_daily, r4_daily)
    s4_daily_aligned = align_htf_to_ltf(prices, df_daily, s4_daily)
    atr_daily_aligned = align_htf_to_ltf(prices, df_daily, atr_daily)
    
    # Load weekly data for trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 10:
        return np.zeros(n)
    
    close_weekly = df_weekly['close'].values
    # Weekly EMA(20) for trend
    ema20_weekly = pd.Series(close_weekly).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema20_weekly)
    
    # Main timeframe data (6h)
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if NaN in critical values
        if (np.isnan(ema20_weekly_aligned[i]) or np.isnan(atr_daily_aligned[i]) or 
            np.isnan(pivot_daily_aligned[i]) or np.isnan(r1_daily_aligned[i]) or 
            np.isnan(s1_daily_aligned[i]) or np.isnan(r4_daily_aligned[i]) or 
            np.isnan(s4_daily_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema20_weekly = ema20_weekly_aligned[i]
        atr_daily = atr_daily_aligned[i]
        pivot = pivot_daily_aligned[i]
        r1 = r1_daily_aligned[i]
        s1 = s1_daily_aligned[i]
        r4 = r4_daily_aligned[i]
        s4 = s4_daily_aligned[i]
        vol_current = volume[i]
        
        # Trend filter: weekly EMA determines bias
        weekly_uptrend = price > ema20_weekly
        weekly_downtrend = price < ema20_weekly
        
        # Volume filter: current volume > 1.5x 20-period average
        vol_ma_recent = np.mean(volume[max(0, i-20):i]) if i >= 20 else volume[i]
        vol_ok = vol_current > 1.5 * vol_ma_recent
        
        # Distance from pivot (for breakout strength)
        dist_from_pivot = abs(price - pivot) / pivot if pivot != 0 else 0
        
        if position == 0:
            # Long breakout: price above R4 with volume in uptrend
            if weekly_uptrend and price > r4 and vol_ok and dist_from_pivot > 0.01:
                signals[i] = 0.25
                position = 1
            # Short breakdown: price below S4 with volume in downtrend
            elif weekly_downtrend and price < s4 and vol_ok and dist_from_pivot > 0.01:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price drops below R1 or trend reverses
            if price < r1 or not weekly_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises above S1 or trend reverses
            if price > s1 or not weekly_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_1w_1d_Pivot_R4S4_Breakout_VolumeTrendFilter_v1"
timeframe = "6h"
leverage = 1.0