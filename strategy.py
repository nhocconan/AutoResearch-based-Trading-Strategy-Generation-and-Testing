#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data for weekly pivot calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate weekly pivot points from daily data
    # Group daily data into weeks (Monday to Friday) - simplified approach using 5-day periods
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly high, low, close using 5-day rolling window (approximation of weekly)
    # Since we don't have actual weekly data, we'll use 5-day aggregates
    window = 5
    weekly_high = pd.Series(high_1d).rolling(window=window, min_periods=window).max().values
    weekly_low = pd.Series(low_1d).rolling(window=window, min_periods=window).min().values
    weekly_close = pd.Series(close_1d).rolling(window=window, min_periods=window).last().values
    
    # Calculate weekly pivot points (using previous week's data to avoid look-ahead)
    # Shift by 1 week (5 periods) to use previous week's data
    pp = (np.roll(weekly_high, window) + np.roll(weekly_low, window) + np.roll(weekly_close, window)) / 3.0
    r1 = 2 * pp - np.roll(weekly_low, window)
    s1 = 2 * pp - np.roll(weekly_high, window)
    r2 = pp + (np.roll(weekly_high, window) - np.roll(weekly_low, window))
    s2 = pp - (np.roll(weekly_high, window) - np.roll(weekly_low, window))
    r3 = pp + 2 * (np.roll(weekly_high, window) - np.roll(weekly_low, window))
    s3 = pp - 2 * (np.roll(weekly_high, window) - np.roll(weekly_low, window))
    
    # Align weekly pivot levels to 6h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # 6h timeframe indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 6h EMA20 for trend confirmation
    ema20_6h = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Volume spike detection (20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or np.isnan(ema20_6h[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        pp_val = pp_aligned[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        r2_val = r2_aligned[i]
        s2_val = s2_aligned[i]
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        ema20_val = ema20_6h[i]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        price = close[i]
        
        # Volume filter: require volume above average to avoid low-volume false signals
        vol_filter = vol > vol_ma
        
        if position == 0:
            # Long: price breaks above R1 with volume and above EMA20
            if price > r1_val and vol_filter and price > ema20_val:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume and below EMA20
            elif price < s1_val and vol_filter and price < ema20_val:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit on breakdown below S1 or below EMA20
                if price < s1_val or price < ema20_val:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit on breakout above R1 or above EMA20
                if price > r1_val or price > ema20_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_WeeklyPivot_R1_S1_Breakout_EMA20_Volume"
timeframe = "6h"
leverage = 1.0