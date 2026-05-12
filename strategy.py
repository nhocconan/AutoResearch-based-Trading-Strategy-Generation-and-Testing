#!/usr/bin/env python3
# 1d_Camarilla_R3_S3_Breakout_1wTrend_VolumeFilter
# Hypothesis: On daily timeframe, enter long when price breaks above Camarilla R3 with price > weekly EMA50 and volume > 1.5x 20-day MA.
# Enter short when price breaks below Camarilla S3 with price < weekly EMA50 and volume > 1.5x 20-day MA.
# Exit when price crosses back below R3 (longs) or above S3 (shorts).
# Uses weekly EMA50 for trend filter to avoid counter-trend trades.
# Targets 15-25 trades/year for low fee drag and works in both bull and bear markets by fading extreme daily levels with institutional levels.

name = "1d_Camarilla_R3_S3_Breakout_1wTrend_VolumeFilter"
timeframe = "1d"
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
    
    # Calculate Camarilla levels from previous day (using daily data)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Previous day's OHLC for Camarilla calculation
    prev_high = np.roll(daily_high, 1)
    prev_low = np.roll(daily_low, 1)
    prev_close = np.roll(daily_close, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Camarilla R3 and S3 levels
    camarilla_range = prev_high - prev_low
    r3 = prev_close + camarilla_range * 1.1 / 4
    s3 = prev_close - camarilla_range * 1.1 / 4
    
    # Weekly EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    weekly_close = df_1w['close'].values
    weekly_ema50 = pd.Series(weekly_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume confirmation: 20-day moving average on daily data
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align all to daily timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    weekly_ema50_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(weekly_ema50_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        weekly_trend = weekly_ema50_aligned[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # LONG: Price breaks above R3 with price > weekly EMA50 and volume > 1.5x MA
            if close[i] > r3_val and close[i] > weekly_trend and volume[i] > vol_ma_val * 1.5:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S3 with price < weekly EMA50 and volume > 1.5x MA
            elif close[i] < s3_val and close[i] < weekly_trend and volume[i] > vol_ma_val * 1.5:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses back below R3 (failed breakout)
            if close[i] < r3_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses back above S3 (failed breakout)
            if close[i] > s3_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals