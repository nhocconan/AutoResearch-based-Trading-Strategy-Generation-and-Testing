#!/usr/bin/env python3
"""
1d_Weekly_Camarilla_R3S3_Breakout_Trend_Volume_v1
Hypothesis: Uses weekly Camarilla pivot levels to determine trend direction, with daily breakout of R3/S3 levels for entry.
In bullish weekly trend (price above weekly R3), look for long entries when daily price breaks above daily R3 with volume confirmation.
In bearish weekly trend (price below weekly S3), look for short entries when daily price breaks below daily S3 with volume confirmation.
Volume confirmation requires current volume > 1.5 * 20-day average volume.
Designed for low trade frequency (~10-20 trades/year) by requiring weekly trend alignment and daily breakout with volume.
Works in both bull and bear markets by following the weekly trend direction.
"""

name = "1d_Weekly_Camarilla_R3S3_Breakout_Trend_Volume_v1"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data for Camarilla pivots and trend
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 2:
        return np.zeros(n)
    
    # Daily OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- Weekly Camarilla Pivot Levels (R3, S3) ---
    # Calculate from previous weekly bar
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    # Pivot point = (H + L + C) / 3
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3
    
    # R3 = Pivot + 2*(H - L)
    # S3 = Pivot - 2*(H - L)
    weekly_r3 = weekly_pivot + 2 * (weekly_high - weekly_low)
    weekly_s3 = weekly_pivot - 2 * (weekly_high - weekly_low)
    
    # Align weekly levels to daily (weekly levels update only after weekly bar closes)
    weekly_r3_aligned = align_htf_to_ltf(prices, df_weekly, weekly_r3)
    weekly_s3_aligned = align_htf_to_ltf(prices, df_weekly, weekly_s3)
    
    # --- Weekly Trend: Price above/below weekly R3/S3 ---
    weekly_bullish = close > weekly_r3_aligned  # Price above weekly R3 = bullish trend
    weekly_bearish = close < weekly_s3_aligned  # Price below weekly S3 = bearish trend
    
    # --- Daily Camarilla Pivot Levels (R3, S3) for entry ---
    # Calculate from previous daily bar
    daily_high = np.roll(high, 1)
    daily_low = np.roll(low, 1)
    daily_close = np.roll(close, 1)
    # First bar: use current values (no previous)
    daily_high[0] = high[0]
    daily_low[0] = low[0]
    daily_close[0] = close[0]
    
    daily_pivot = (daily_high + daily_low + daily_close) / 3
    daily_r3 = daily_pivot + 2 * (daily_high - daily_low)
    daily_s3 = daily_pivot - 2 * (daily_high - daily_low)
    
    # --- Volume Confirmation ---
    # Average volume over 20 days
    vol_series = pd.Series(volume)
    avg_volume = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_confirmation = volume > (1.5 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 40  # Need 20 for volume avg + extra
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(weekly_r3_aligned[i]) or np.isnan(weekly_s3_aligned[i]) or
            np.isnan(daily_r3[i]) or np.isnan(daily_s3[i]) or
            np.isnan(avg_volume[i])):
            # Maintain position if valid, otherwise flat
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Look for entries aligned with weekly trend
            if weekly_bullish[i]:
                # Bullish weekly trend: look for long on daily R3 breakout with volume
                if close[i] > daily_r3[i] and volume_confirmation[i]:
                    signals[i] = 0.25
                    position = 1
            elif weekly_bearish[i]:
                # Bearish weekly trend: look for short on daily S3 breakout with volume
                if close[i] < daily_s3[i] and volume_confirmation[i]:
                    signals[i] = -0.25
                    position = -1
        else:
            # Exit conditions: price returns to daily pivot or weekly trend changes
            if position == 1:
                # Exit long: price below daily pivot or weekly trend turns bearish
                exit_signal = (close[i] < daily_pivot[i]) or (~weekly_bullish[i])
                if exit_signal:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price above daily pivot or weekly trend turns bullish
                exit_signal = (close[i] > daily_pivot[i]) or (~weekly_bearish[i])
                if exit_signal:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals