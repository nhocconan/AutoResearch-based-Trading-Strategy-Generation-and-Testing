#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h timeframe with 1d Elder Ray (Bull/Bear Power) and weekly regime filter.
# Elder Ray measures bull power (high - EMA13) and bear power (low - EMA13).
# Long: Bull Power > 0 AND Bear Power < 0 AND weekly EMA21 trend up (close > EMA21).
# Short: Bear Power < 0 AND Bull Power < 0 AND weekly EMA21 trend down (close < EMA21).
# Uses 13-period EMA for power calculation, 21-period weekly EMA for trend filter.
# Volume confirmation: current volume > 1.3x average volume (20-period).
# Position size: 0.25 (25%).
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for Elder Ray calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate EMA13 for 1d data
    ema13 = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 13:
        ema13[12] = np.mean(close_1d[:13])
        for i in range(13, len(close_1d)):
            ema13[i] = (close_1d[i] * 2 / (13 + 1)) + (ema13[i-1] * (12 / (13 + 1)))
    
    # Calculate Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = np.full(len(close_1d), np.nan)
    bear_power = np.full(len(close_1d), np.nan)
    for i in range(len(close_1d)):
        if not np.isnan(ema13[i]):
            bull_power[i] = high_1d[i] - ema13[i]
            bear_power[i] = low_1d[i] - ema13[i]
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 25:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate EMA21 for weekly data
    ema21_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 21:
        ema21_1w[20] = np.mean(close_1w[:21])
        for i in range(21, len(close_1w)):
            ema21_1w[i] = (close_1w[i] * 2 / (21 + 1)) + (ema21_1w[i-1] * (20 / (21 + 1)))
    
    # Average volume (20-period) for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    # Align 1d Elder Ray and weekly EMA21 to 6h
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    ema21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema21_1w)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if any required data is not ready
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(ema21_1w_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        bp = bull_power_aligned[i]
        be = bear_power_aligned[i]
        ema21 = ema21_1w_aligned[i]
        
        # Volume confirmation: current volume > 1.3x average volume
        volume_confirm = vol > 1.3 * avg_vol
        
        if position == 0:
            # Long: Bull Power > 0 AND Bear Power < 0 AND weekly trend up + volume confirmation
            if (bp > 0 and be < 0 and price > ema21 and volume_confirm):
                position = 1
                signals[i] = position_size
            # Short: Bear Power < 0 AND Bull Power < 0 AND weekly trend down + volume confirmation
            elif (be < 0 and bp < 0 and price < ema21 and volume_confirm):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Bear Power >= 0 (bullish momentum fading)
            if be >= 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Bull Power <= 0 (bearish momentum fading)
            if bp <= 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1d_1w_Elder_Ray_Weekly_Trend"
timeframe = "6h"
leverage = 1.0