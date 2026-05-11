#!/usr/bin/env python3
"""
6h_Weekly_Camarilla_R4S4_Breakout_Trend_v1
Hypothesis: Uses weekly Camarilla pivot levels (R4/S4) for breakout entries aligned with 12h trend.
In strong trends (12h EMA50 slope > 0 for long, < 0 for short), we take breakout positions when price
closes beyond weekly R4 (long) or S4 (short). Uses volume confirmation to avoid false breakouts.
Designed for low trade frequency (~15-25 trades/year) by requiring weekly levels and trend alignment.
Works in both bull and bear markets by following the 12h trend direction.
"""

name = "6h_Weekly_Camarilla_R4S4_Breakout_Trend_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data for Camarilla levels
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 5:
        return np.zeros(n)
    
    # Get 12h data for trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 6h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- Calculate Weekly Camarilla Levels (R4, S4) ---
    # Using previous week's OHLC
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    # Calculate pivot and ranges
    pivot = (weekly_high + weekly_low + weekly_close) / 3
    range_val = weekly_high - weekly_low
    
    # Camarilla levels
    R4 = pivot + (range_val * 1.1 / 2)
    S4 = pivot - (range_val * 1.1 / 2)
    
    # Align weekly levels to 6h
    R4_aligned = align_htf_to_ltf(prices, df_weekly, R4)
    S4_aligned = align_htf_to_ltf(prices, df_weekly, S4)
    
    # --- 12h EMA50 for Trend ---
    close_12h = df_12h['close'].values
    ema_50 = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    
    # EMA slope (trend direction)
    ema_slope = np.diff(ema_50_aligned, prepend=ema_50_aligned[0])
    
    # --- Volume Confirmation ---
    # Volume ratio: current volume vs 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)  # Avoid division by zero
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(R4_aligned[i]) or np.isnan(S4_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(ema_slope[i]) or
            np.isnan(vol_ratio[i])):
            # Maintain position if valid, otherwise flat
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Look for breakout signals aligned with trend
            # Long: price closes above weekly R4 with upward trend and volume confirmation
            if (close[i] > R4_aligned[i] and 
                ema_slope[i] > 0 and 
                vol_ratio[i] > 1.5):  # Volume 1.5x average
                signals[i] = 0.25
                position = 1
            # Short: price closes below weekly S4 with downward trend and volume confirmation
            elif (close[i] < S4_aligned[i] and 
                  ema_slope[i] < 0 and 
                  vol_ratio[i] > 1.5):  # Volume 1.5x average
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: price returns to pivot area or trend reverses
            if position == 1:
                # Exit long: price below pivot or trend turns down
                exit_signal = (close[i] < (R4_aligned[i] + S4_aligned[i]) / 2) or (ema_slope[i] < 0)
                if exit_signal:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price above pivot or trend turns up
                exit_signal = (close[i] > (R4_aligned[i] + S4_aligned[i]) / 2) or (ema_slope[i] > 0)
                if exit_signal:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals