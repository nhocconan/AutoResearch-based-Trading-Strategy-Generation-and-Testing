#!/usr/bin/env python3
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
    
    # Get 1d data for weekly pivot levels and trend
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate weekly pivot points using actual weekly aggregation
    # We'll use 7-day rolling window for weekly OHLC (approximation)
    weekly_high = pd.Series(high_1d).rolling(window=7, min_periods=7).max().shift(1).values
    weekly_low = pd.Series(low_1d).rolling(window=7, min_periods=7).min().shift(1).values
    weekly_close = pd.Series(close_1d).rolling(window=7, min_periods=7).last().shift(1).values
    
    # Weekly pivot point: (H + L + C) / 3
    pp = (weekly_high + weekly_low + weekly_close) / 3
    # Weekly range
    weekly_range = weekly_high - weekly_low
    # Resistance levels
    r1 = 2 * pp - weekly_low
    r2 = pp + weekly_range
    r3 = weekly_high + 2 * (pp - weekly_low)
    # Support levels
    s1 = 2 * pp - weekly_high
    s2 = pp - weekly_range
    s3 = weekly_low - 2 * (weekly_high - pp)
    
    # Align weekly pivot levels to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r3 + weekly_range)  # R4 = R3 + range
    s4_aligned = align_htf_to_ltf(prices, df_1d, s3 - weekly_range)  # S4 = S3 - range
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    # Weekly EMA200 approximation (using 200/5 = 40 periods on weekly)
    ema_200_1w = pd.Series(close_1w).ewm(span=40, adjust=False, min_periods=40).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Volume confirmation: volume > 1.5x average volume (20-period) on 6h
    vol_series = pd.Series(volume)
    avg_vol = vol_series.rolling(window=20, min_periods=20).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(40, 20)  # 40 for weekly EMA, 20 for volume
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(ema_200_1w_aligned[i]) or np.isnan(avg_vol[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        if position == 0:
            # Long: price breaks above R4 AND above weekly EMA200 with volume
            if price > r4_aligned[i] and price > ema_200_1w_aligned[i] and vol > 1.5 * avg_vol[i]:
                position = 1
                signals[i] = position_size
            # Short: price breaks below S4 AND below weekly EMA200 with volume
            elif price < s4_aligned[i] and price < ema_200_1w_aligned[i] and vol > 1.5 * avg_vol[i]:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below R3 or below weekly EMA200
            if price < r3_aligned[i] or price < ema_200_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above S3 or above weekly EMA200
            if price > s3_aligned[i] or price > ema_200_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1w_1d_Camarilla_Trend_Filter"
timeframe = "6h"
leverage = 1.0