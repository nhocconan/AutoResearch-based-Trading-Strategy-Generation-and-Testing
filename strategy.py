#!/usr/bin/env python3
# 6H_1D_Camarilla_R3_S3_Breakout_1dTrend_Volume
# Hypothesis: Use daily Camarilla R3/S3 levels as breakout levels with daily trend filter (EMA34) and volume confirmation.
# Enters long when price breaks above R3 in bullish daily trend with volume > 1.5x average.
# Enters short when price breaks below S3 in bearish daily trend with volume > 1.5x average.
# Uses tight entries to avoid overtrading, works in bull/bear by following daily trend direction.
# Target: 15-30 trades/year per symbol.

name = "6H_1D_Camarilla_R3_S3_Breakout_1dTrend_Volume"
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
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Daily EMA34 for trend
    close_1d_series = pd.Series(close_1d)
    ema34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Daily Camarilla levels (using prior day's OHLC)
    # Typical price = (H + L + C) / 3
    # Range = H - L
    # R3 = Close + 1.1 * (High - Low) * 1.1/2
    # S3 = Close - 1.1 * (High - Low) * 1.1/2
    typical_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    r3_1d = close_1d + 1.1 * range_1d * 1.1 / 2.0
    s3_1d = close_1d - 1.1 * range_1d * 1.1 / 2.0
    
    # Prepend NaN for alignment (current day's levels use prior day's data)
    r3_1d = np.concatenate([[np.nan], r3_1d])
    s3_1d = np.concatenate([[np.nan], s3_1d])
    
    # Trend: bullish if close > EMA34, bearish if close < EMA34
    bullish_trend = close_1d > ema34_1d
    bearish_trend = close_1d < ema34_1d
    
    # Volume average (20-period)
    vol_series = pd.Series(volume)
    vol_avg = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_threshold = 1.5  # volume must be > 1.5x average
    
    # Align to 6h
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    bullish_aligned = align_htf_to_ltf(prices, df_1d, bullish_trend.astype(float))
    bearish_aligned = align_htf_to_ltf(prices, df_1d, bearish_trend.astype(float))
    vol_avg_aligned = align_htf_to_ltf(prices, df_1d, vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(bullish_aligned[i]) or np.isnan(bearish_aligned[i]) or
            np.isnan(vol_avg_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        bullish = bullish_aligned[i] > 0.5
        bearish = bearish_aligned[i] > 0.5
        vol_ok = volume[i] > vol_avg_aligned[i] * vol_threshold
        
        if position == 0:
            # Enter long: bullish trend + price breaks above R3 + volume confirmation
            if bullish and close[i] > r3_aligned[i] and vol_ok:
                signals[i] = 0.25
                position = 1
            # Enter short: bearish trend + price breaks below S3 + volume confirmation
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