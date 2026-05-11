#!/usr/bin/env python3
# 1d_Weekly_Trend_Riding
# Hypothesis: Capture weekly trends with daily entries using 1-week SMA trend filter and daily Donchian breakout for timing. Uses volume confirmation to avoid false breakouts. Designed for low trade frequency (<25/year) to minimize fee drag. Works in bull markets by riding uptrends and in bear markets by capturing short-term reversals within the weekly trend context.

name = "1d_Weekly_Trend_Riding"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data for trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 20:
        return np.zeros(n)
    
    # Daily OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- Weekly trend: 20-period SMA ---
    weekly_close = df_weekly['close'].values
    weekly_sma = np.full(len(weekly_close), np.nan)
    for i in range(20, len(weekly_close)):
        weekly_sma[i] = np.mean(weekly_close[i-20:i])
    
    # Align weekly SMA to daily timeframe
    weekly_sma_aligned = align_htf_to_ltf(prices, df_weekly, weekly_sma)
    
    # --- Daily Donchian channel (20-period) ---
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    for i in range(20, n):
        highest_high[i] = np.max(high[i-20:i])
        lowest_low[i] = np.min(low[i-20:i])
    
    # --- Volume confirmation: volume > 1.5x 20-day average ---
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # need enough for Donchian and volume MA
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(highest_high[i]) or
            np.isnan(lowest_low[i]) or
            np.isnan(vol_ma[i]) or
            np.isnan(weekly_sma_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Weekly trend filter
        weekly_uptrend = close[i] > weekly_sma_aligned[i]
        weekly_downtrend = close[i] < weekly_sma_aligned[i]
        
        # Volume spike condition
        vol_spike = volume[i] > vol_ma[i] * 1.5
        
        if position == 0:
            # Long: weekly uptrend + price breaks above Donchian high + volume spike
            if weekly_uptrend and close[i] > highest_high[i] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: weekly downtrend + price breaks below Donchian low + volume spike
            elif weekly_downtrend and close[i] < lowest_low[i] and vol_spike:
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Exit long: price falls below Donchian low OR weekly trend turns down
                if close[i] < lowest_low[i] or not weekly_uptrend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price rises above Donchian high OR weekly trend turns up
                if close[i] > highest_high[i] or not weekly_downtrend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals