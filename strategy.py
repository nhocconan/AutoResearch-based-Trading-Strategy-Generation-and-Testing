#!/usr/bin/env python3
name = "6h_WeeklyPivot_R3S3_Breakout_TrendVolume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for weekly pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate weekly pivot points using last week's high/low/close
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous week's HLC (5-day lookback for weekly)
    prev_week_high = pd.Series(high_1d).rolling(window=5, min_periods=5).max().shift(5).values
    prev_week_low = pd.Series(low_1d).rolling(window=5, min_periods=5).min().shift(5).values
    prev_week_close = pd.Series(close_1d).shift(5).values
    
    # Weekly pivot point
    pivot = (prev_week_high + prev_week_low + prev_week_close) / 3.0
    
    # Weekly support/resistance levels
    r1 = 2 * pivot - prev_week_low
    s1 = 2 * pivot - prev_week_high
    r2 = pivot + (prev_week_high - prev_week_low)
    s2 = pivot - (prev_week_high - prev_week_low)
    r3 = prev_week_high + 2 * (pivot - prev_week_low)
    s3 = prev_week_low - 2 * (prev_week_high - pivot)
    r4 = prev_week_high + 3 * (pivot - prev_week_low)
    s4 = prev_week_low - 3 * (prev_week_high - pivot)
    
    # Align weekly levels to 6h
    pivot_6h = align_htf_to_ltf(prices, df_1d, pivot)
    r3_6h = align_htf_to_ltf(prices, df_1d, r3)
    s3_6h = align_htf_to_ltf(prices, df_1d, s3)
    r4_6h = align_htf_to_ltf(prices, df_1d, r4)
    s4_6h = align_htf_to_ltf(prices, df_1d, s4)
    
    # Trend filter: 20-period EMA on daily (aligned)
    ema_20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_6h = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    # Volume filter: 20-period average on 6h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.divide(volume, vol_ma, out=np.ones_like(volume), where=vol_ma!=0)
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_6h[i]) or np.isnan(r3_6h[i]) or np.isnan(s3_6h[i]) or 
            np.isnan(r4_6h[i]) or np.isnan(s4_6h[i]) or np.isnan(ema_20_6h[i]) or 
            np.isnan(vol_ratio[i])):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume threshold - avoid low-volume false breakouts
        volume_surge = vol_ratio[i] > 1.5
        
        if position == 0:
            # Long: Price breaks above weekly R3 with volume AND above daily EMA20 (uptrend)
            if (close[i] > r3_6h[i] and volume_surge and close[i] > ema_20_6h[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below weekly S3 with volume AND below daily EMA20 (downtrend)
            elif (close[i] < s3_6h[i] and volume_surge and close[i] < ema_20_6h[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to weekly pivot or opposite S3/R3 level
            if position == 1:
                # Exit long: price returns to pivot or breaks below S3
                if (close[i] < pivot_6h[i]) or (close[i] < s3_6h[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price returns to pivot or breaks above R3
                if (close[i] > pivot_6h[i]) or (close[i] > r3_6h[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals