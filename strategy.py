#!/usr/bin/env python3
"""
4h Daily Pivot R1/S1 Breakout with Volume Spike and EMA Trend Filter
Hypothesis: Daily pivot levels (R1/S1) act as strong support/resistance. 
Breaking above R1 with volume and above daily EMA50 indicates bullish momentum.
Breaking below S1 with volume and below daily EMA50 indicates bearish momentum.
Daily EMA trend filter prevents counter-trend trades. Designed for 20-40 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for pivot and trend filter (once before loop)
    df_d = get_htf_data(prices, '1d')
    
    # Calculate daily high, low, close for pivot levels
    daily_high = df_d['high'].values
    daily_low = df_d['low'].values
    daily_close = df_d['close'].values
    
    # Calculate daily pivot: P = (H+L+C)/3
    daily_pivot = (daily_high + daily_low + daily_close) / 3.0
    daily_range = daily_high - daily_low
    # Daily R1 = P + (H-L) = P + range
    daily_r1 = daily_pivot + daily_range
    # Daily S1 = P - (H-L) = P - range
    daily_s1 = daily_pivot - daily_range
    
    # Calculate daily EMA50 for trend filter
    ema_50_d = pd.Series(daily_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align daily levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_d, daily_r1)
    s1_aligned = align_htf_to_ltf(prices, df_d, daily_s1)
    pivot_aligned = align_htf_to_ltf(prices, df_d, daily_pivot)
    ema_50_d_aligned = align_htf_to_ltf(prices, df_d, ema_50_d)
    
    # Volume spike detection (1.5x 30-period average)
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 100  # need enough history for calculations
    
    for i in range(start_idx, n):
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or
            np.isnan(pivot_aligned[i]) or
            np.isnan(ema_50_d_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        r1_level = r1_aligned[i]
        s1_level = s1_aligned[i]
        pivot_level = pivot_aligned[i]
        ema_50 = ema_50_d_aligned[i]
        
        if position == 0:
            # Long: break above daily R1 with volume spike and above daily EMA50
            if (price > r1_level and volume_spike[i] and price > ema_50):
                signals[i] = 0.25
                position = 1
            # Short: break below daily S1 with volume spike and below daily EMA50
            elif (price < s1_level and volume_spike[i] and price < ema_50):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long position management
            signals[i] = 0.25
            # Exit: price returns to daily pivot or below daily EMA50 (trend change)
            if price <= pivot_level or price < ema_50:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position management
            signals[i] = -0.25
            # Exit: price returns to daily pivot or above daily EMA50 (trend change)
            if price >= pivot_level or price > ema_50:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_DailyPivot_R1S1_Breakout_VolumeSpike_1dEMA50"
timeframe = "4h"
leverage = 1.0