#!/usr/bin/env python3
# 6h_1d_Camarilla_R3S3_Breakout_Trend
# Hypothesis: In strong daily trends, price breaks through Camarilla R3/S3 levels with volume confirmation,
# indicating trend continuation. Uses 1d Camarilla levels (R3/S3) and 1d EMA50 for trend filter.
# Works in bull/bear by following daily trend direction. Target: 15-25 trades/year per symbol.

name = "6h_1d_Camarilla_R3S3_Breakout_Trend"
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
    volume = prices['volume'].values
    
    # Get daily data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for previous day
    # Camarilla: R3 = C + (H-L)*1.1/2, S3 = C - (H-L)*1.1/2
    # Using previous day's values to avoid look-ahead
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d = np.roll(close_1d, 1)
    
    # Set first value to NaN since we don't have previous day
    prev_high_1d[0] = np.nan
    prev_low_1d[0] = np.nan
    prev_close_1d[0] = np.nan
    
    rang = prev_high_1d - prev_low_1d
    r3 = prev_close_1d + rang * 1.1 / 2
    s3 = prev_close_1d - rang * 1.1 / 2
    
    # Daily EMA50 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Trend: bullish if close > EMA50, bearish if close < EMA50
    bullish_trend = close_1d > ema50_1d
    bearish_trend = close_1d < ema50_1d
    
    # Align to 6h
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    bullish_aligned = align_htf_to_ltf(prices, df_1d, bullish_trend.astype(float))
    bearish_aligned = align_htf_to_ltf(prices, df_1d, bearish_trend.astype(float))
    
    # Volume filter: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(ema50_aligned[i]) or
            np.isnan(bullish_aligned[i]) or np.isnan(bearish_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        bullish = bullish_aligned[i] > 0.5
        bearish = bearish_aligned[i] > 0.5
        vol_ok = volume[i] > vol_ma[i]  # Volume above average
        
        if position == 0:
            # Enter long: bullish trend + price breaks above R3 + volume
            if bullish and close[i] > r3_aligned[i] and vol_ok:
                signals[i] = 0.25
                position = 1
            # Enter short: bearish trend + price breaks below S3 + volume
            elif bearish and close[i] < s3_aligned[i] and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: bearish trend or price falls back below R3
            if bearish or close[i] < r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: bullish trend or price rises back above S3
            if bullish or close[i] > s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals