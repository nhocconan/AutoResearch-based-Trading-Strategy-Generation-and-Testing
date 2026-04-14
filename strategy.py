# -*- coding: utf-8 -*-
# -*- mode: python; python-indent: 4; -*-

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h timeframe with weekly pivot levels (calculated from prior week's OHLC) and EMA(50) trend filter.
# Long when price breaks above weekly R3 with price > EMA50 and volume > 1.5x average volume.
# Short when price breaks below weekly S3 with price < EMA50 and volume > 1.5x average volume.
# Exit when price crosses back below weekly R2 (for longs) or above weekly S2 (for shorts).
# Weekly pivots use prior week's data to avoid look-ahead. Volume filter uses 20-period average.
# This structure aims to capture strong trending moves with institutional interest at extreme pivot levels,
# working in both bull (breakouts) and bear (breakdowns) markets by using symmetric long/short logic.
# Target: 50-150 total trades over 4 years (~12-37/year) to avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot points and trend filter
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA(50) for trend filter
    close_1w_series = pd.Series(close_1w)
    ema_50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate weekly pivot points (using prior week's OHLC)
    prev_week_high = np.roll(high_1w, 1)
    prev_week_low = np.roll(low_1w, 1)
    prev_week_close = np.roll(close_1w, 1)
    prev_week_high[0] = np.nan
    prev_week_low[0] = np.nan
    prev_week_close[0] = np.nan
    
    # Weekly pivot point
    pp = (prev_week_high + prev_week_low + prev_week_close) / 3
    # Weekly resistance and support levels
    r1 = 2 * pp - prev_week_low
    s1 = 2 * pp - prev_week_high
    r2 = pp + (prev_week_high - prev_week_low)
    s2 = pp - (prev_week_high - prev_week_low)
    r3 = pp + 2 * (prev_week_high - prev_week_low)
    s3 = pp - 2 * (prev_week_high - prev_week_low)
    
    # Align weekly pivot levels to weekly timeframe (using prior week's data, no shift needed)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: volume > 1.5x average volume (20-period)
    vol_series = pd.Series(volume)
    avg_vol = vol_series.rolling(window=20, min_periods=20).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(50, 20)  # for 50-period EMA and 20-period volume average
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(r2_aligned[i]) or
            np.isnan(s2_aligned[i]) or np.isnan(avg_vol[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        if position == 0:
            # Long: price breaks above weekly R3 AND above weekly EMA50 with volume filter
            if (price > r3_aligned[i] and price > ema_50_1w_aligned[i] and 
                vol > 1.5 * avg_vol[i]):
                position = 1
                signals[i] = position_size
            # Short: price breaks below weekly S3 AND below weekly EMA50 with volume filter
            elif (price < s3_aligned[i] and price < ema_50_1w_aligned[i] and 
                  vol > 1.5 * avg_vol[i]):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below weekly R2
            if price < r2_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses above weekly S2
            if price > s2_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_Weekly_Pivot_EMA_Volume"
timeframe = "6h"
leverage = 1.0