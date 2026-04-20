#!/usr/bin/env python3
# 1d_WeeklyDonchian_Breakout_Volume_TrendFilter_V1
# Hypothesis: Weekly Donchian channel breakouts (20-period) capture major trends, with volume confirmation and daily trend filter to reduce whipsaws.
# Works in both bull and bear markets by only taking breakouts in the direction of the daily EMA50 trend. Weekly timeframe reduces trade frequency.

name = "1d_WeeklyDonchian_Breakout_Volume_TrendFilter_V1"
timeframe = "1d"
leverage = 1.0

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
    
    # Get weekly data ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 20:
        return np.zeros(n)
    
    # Calculate weekly Donchian channels (20-period)
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    
    # Upper band: highest high over last 20 weekly bars
    upper_raw = pd.Series(weekly_high).rolling(window=20, min_periods=20).max().values
    # Lower band: lowest low over last 20 weekly bars
    lower_raw = pd.Series(weekly_low).rolling(window=20, min_periods=20).min().values
    
    # Align to daily timeframe
    upper_aligned = align_htf_to_ltf(prices, df_weekly, upper_raw)
    lower_aligned = align_htf_to_ltf(prices, df_weekly, lower_raw)
    
    # Get daily data for trend filter
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 50:
        return np.zeros(n)
    
    # Daily EMA50 for trend filter
    daily_close = df_daily['close'].values
    ema50_raw = pd.Series(daily_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_aligned = align_htf_to_ltf(prices, df_daily, ema50_raw)
    
    # Volume filter: volume > 1.5x 20-day average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Ensure all indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or
            np.isnan(ema50_aligned[i]) or np.isnan(volume_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above weekly upper band + volume + above daily EMA50 (uptrend)
            if close[i] > upper_aligned[i] and volume_filter[i] and close[i] > ema50_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly lower band + volume + below daily EMA50 (downtrend)
            elif close[i] < lower_aligned[i] and volume_filter[i] and close[i] < ema50_aligned[i]:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below weekly lower band (trend reversal)
            if close[i] < lower_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above weekly upper band (trend reversal)
            if close[i] > upper_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals