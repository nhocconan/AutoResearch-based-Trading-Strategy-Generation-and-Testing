#!/usr/bin/env python3
# 12H_1W_Camarilla_R3_S3_Breakout_Trend_Volume
# Hypothesis: Breakout at weekly Camarilla R3/S3 levels with trend alignment and volume confirmation.
# Long when: price breaks above weekly R3, weekly trend up (price > weekly EMA50), and volume > 1.5x average.
# Short when: price breaks below weekly S3, weekly trend down (price < weekly EMA50), and volume > 1.5x average.
# Uses daily EMA200 as filter: only trade long if price > daily EMA200, short if price < daily EMA200.
# Works in bull/bear by following weekly trend and using volume to confirm institutional interest.
# Target: 15-30 trades/year per symbol.

name = "12H_1W_Camarilla_R3_S3_Breakout_Trend_Volume"
timeframe = "12h"
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
    
    # Weekly data for Camarilla levels and trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Camarilla levels (using previous week's range)
    # R3 = close + 1.1 * (high - low)
    # S3 = close - 1.1 * (high - low)
    prev_high = np.roll(high_1w, 1)
    prev_low = np.roll(low_1w, 1)
    prev_close = np.roll(close_1w, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    r3 = prev_close + 1.1 * (prev_high - prev_low)
    s3 = prev_close - 1.1 * (prev_high - prev_low)
    
    # Weekly EMA50 for trend
    close_1w_series = pd.Series(close_1w)
    ema50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    weekly_uptrend = close_1w > ema50_1w
    weekly_downtrend = close_1w < ema50_1w
    
    # Align weekly data to 12h
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    weekly_uptrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_uptrend.astype(float))
    weekly_downtrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_downtrend.astype(float))
    
    # Daily EMA200 filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    daily_filter_long = close_1d > ema200_1d
    daily_filter_short = close_1d < ema200_1d
    
    daily_long_aligned = align_htf_to_ltf(prices, df_1d, daily_filter_long.astype(float))
    daily_short_aligned = align_htf_to_ltf(prices, df_1d, daily_filter_short.astype(float))
    
    # Volume average (20-period)
    volume_s = pd.Series(volume)
    vol_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(weekly_uptrend_aligned[i]) or np.isnan(weekly_downtrend_aligned[i]) or
            np.isnan(daily_long_aligned[i]) or np.isnan(daily_short_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_confirm = vol_ratio > 1.5
        
        weekly_up = weekly_uptrend_aligned[i] > 0.5
        weekly_down = weekly_downtrend_aligned[i] > 0.5
        daily_long_ok = daily_long_aligned[i] > 0.5
        daily_short_ok = daily_short_aligned[i] > 0.5
        
        if position == 0:
            # Enter long: weekly uptrend + price > R3 + volume + daily filter
            if weekly_up and close[i] > r3_aligned[i] and volume_confirm and daily_long_ok:
                signals[i] = 0.25
                position = 1
            # Enter short: weekly downtrend + price < S3 + volume + daily filter
            elif weekly_down and close[i] < s3_aligned[i] and volume_confirm and daily_short_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: weekly trend down or price < S3
            if not weekly_up or close[i] < s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: weekly trend up or price > R3
            if not weekly_down or close[i] > r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals