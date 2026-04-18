#!/usr/bin/env python3
"""
1d_Weekly_Pivot_R1S1_Breakout_Volume_Spike_v1
Hypothesis: Weekly Pivot R1/S1 breakouts with volume confirmation and trend filter capture strong moves in both bull and bear markets. Weekly pivots provide institutional reference points; breakouts above R1 or below S1 with volume indicate institutional participation. Trend filter (1w EMA34) ensures we trade with the weekly trend, reducing false signals. Designed for ~15-20 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_htf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for pivot points and trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Weekly OHLC for pivot calculation
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Calculate weekly pivot points: P = (H+L+C)/3
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    # R1 = 2*P - L, S1 = 2*P - H
    weekly_r1 = 2 * weekly_pivot - weekly_low
    weekly_s1 = 2 * weekly_pivot - weekly_high
    
    # Align weekly levels to daily
    r1_daily = align_htf_to_ltf(prices, df_1w, weekly_r1)
    s1_daily = align_htf_to_ltf(prices, df_1w, weekly_s1)
    pivot_daily = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # Weekly EMA34 for trend filter
    weekly_ema34 = pd.Series(weekly_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    weekly_ema34_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema34)
    
    # Daily volume spike: >2.0x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 40
    
    for i in range(start_idx, n):
        if (np.isnan(r1_daily[i]) or np.isnan(s1_daily[i]) or 
            np.isnan(weekly_ema34_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        r1 = r1_daily[i]
        s1 = s1_daily[i]
        weekly_ema = weekly_ema34_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: price breaks above R1 with volume, above weekly EMA (uptrend)
            if price > r1 and vol_spike and price > weekly_ema:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume, below weekly EMA (downtrend)
            elif price < s1 and vol_spike and price < weekly_ema:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price returns to pivot or weekly EMA
            if price <= pivot_daily[i] or price <= weekly_ema:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price returns to pivot or weekly EMA
            if price >= pivot_daily[i] or price >= weekly_ema:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Weekly_Pivot_R1S1_Breakout_Volume_Spike_v1"
timeframe = "1d"
leverage = 1.0