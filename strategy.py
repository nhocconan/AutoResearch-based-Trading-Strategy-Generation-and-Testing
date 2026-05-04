#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Weekly Camarilla Pivot Breakout with 1d Volume Spike and 1w EMA50 Trend Filter
# Weekly Camarilla pivot levels (R3, S3) provide high-probability breakout zones.
# Breakout above R3 or below S3 with volume spike and alignment with weekly EMA50 trend.
# Designed for 12-37 trades/year on 6h to minimize fee drag while capturing strong weekly momentum.
# Works in bull markets via long breakouts in uptrend and bear markets via short breakouts in downtrend.

name = "6h_WeeklyCamarilla_R3S3_Breakout_1wEMA50_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for HTF pivot and trend - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly Camarilla pivot levels (based on prior week)
    # Standard Camarilla: R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low)
    # S3 = close - 1.1*(high-low), S4 = close - 1.5*(high-low)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate pivot levels for prior week (shifted by 1 to avoid look-ahead)
    pivot_high = np.roll(high_1w, 1)
    pivot_low = np.roll(low_1w, 1)
    pivot_close = np.roll(close_1w, 1)
    pivot_high[0] = np.nan  # First value invalid due to roll
    pivot_low[0] = np.nan
    pivot_close[0] = np.nan
    
    # Camarilla levels from prior week
    r3 = pivot_close + 1.1 * (pivot_high - pivot_low)
    s3 = pivot_close - 1.1 * (pivot_high - pivot_low)
    
    # Weekly EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF weekly data to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Get daily data for volume spike confirmation - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    vol_ema_20_1d = pd.Series(volume_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ema_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ema_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ema_20_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current 6h volume > 2x daily average volume (scaled)
        # Approximate: daily volume / 4 = expected 6h volume (since 4x 6h in 1d)
        expected_6h_volume = vol_ema_20_aligned[i] / 4.0
        volume_spike = volume[i] > (expected_6h_volume * 2.0)
        
        if position == 0:
            # Long breakout: price > R3 with volume spike and weekly uptrend
            if (close[i] > r3_aligned[i] and 
                volume_spike and 
                close[i] > ema_50_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short breakdown: price < S3 with volume spike and weekly downtrend
            elif (close[i] < s3_aligned[i] and 
                  volume_spike and 
                  close[i] < ema_50_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below R3 or weekly trend turns down
            if (close[i] < r3_aligned[i] or 
                close[i] < ema_50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above S3 or weekly trend turns up
            if (close[i] > s3_aligned[i] or 
                close[i] > ema_50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals