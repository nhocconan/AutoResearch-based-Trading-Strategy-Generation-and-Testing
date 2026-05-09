#!/usr/bin/env python3
# 4h_Camarilla_R3_S3_Breakout_12hEMA50_Trend
# Hypothesis: 4h timeframe with Camarilla pivot levels (R3/S3) breakout and 12h EMA50 trend filter.
# Enters long when price breaks above R3 and 12h EMA50 is rising, short when price breaks below S3 and 12h EMA50 is falling.
# Uses volume confirmation (current volume > 1.5x 20-period average) to filter breakouts.
# Exits when price returns to the Camarilla midpoint (P) or trend reverses.
# Designed for low trade frequency (target: 20-50 trades/year) with size 0.25 to minimize fee drag.
# Works in both bull and bear markets by following the 12h trend direction.

name = "4h_Camarilla_R3_S3_Breakout_12hEMA50_Trend"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_rising = ema_50_12h > np.roll(ema_50_12h, 1)
    ema_50_12h_falling = ema_50_12h < np.roll(ema_50_12h, 1)
    ema_50_12h_rising = np.where(np.isnan(ema_50_12h_rising), False, ema_50_12h_rising)
    ema_50_12h_falling = np.where(np.isnan(ema_50_12h_falling), False, ema_50_12h_falling)
    ema_50_12h_rising_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h_rising)
    ema_50_12h_falling_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h_falling)
    
    # Calculate Camarilla pivot levels from previous 1d
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Previous day's OHLC (use shift by 1 to avoid look-ahead)
    prev_high = np.roll(df_1d['high'], 1)
    prev_low = np.roll(df_1d['low'], 1)
    prev_close = np.roll(df_1d['close'], 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Camarilla calculations
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_hl = prev_high - prev_low
    
    # Resistance and Support levels
    r3 = pivot + (range_hl * 1.1 / 4)
    s3 = pivot - (range_hl * 1.1 / 4)
    # Midpoint (P) for exit
    midpoint = pivot
    
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    midpoint_aligned = align_htf_to_ltf(prices, df_1d, midpoint)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_12h_rising_aligned[i]) or np.isnan(ema_50_12h_falling_aligned[i]) or
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(midpoint_aligned[i]) or
            np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above R3, 12h EMA50 rising, volume confirmation
            if close[i] > r3_aligned[i] and ema_50_12h_rising_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S3, 12h EMA50 falling, volume confirmation
            elif close[i] < s3_aligned[i] and ema_50_12h_falling_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Hold long: 0.25
            signals[i] = 0.25
            # Exit long: price returns to midpoint OR trend turns bearish
            if close[i] <= midpoint_aligned[i] or not ema_50_12h_rising_aligned[i]:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Hold short: -0.25
            signals[i] = -0.25
            # Exit short: price returns to midpoint OR trend turns bullish
            if close[i] >= midpoint_aligned[i] or not ema_50_12h_falling_aligned[i]:
                signals[i] = 0.0
                position = 0
    
    return signals