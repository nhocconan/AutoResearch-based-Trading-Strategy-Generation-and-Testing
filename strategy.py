#!/usr/bin/env python3
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
    
    # Get weekly data for weekly pivot points
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points using prior week's OHLC
    # Weekly pivot = (prev_week_high + prev_week_low + prev_week_close) / 3
    prev_week_high = np.roll(high_1w, 1)
    prev_week_low = np.roll(low_1w, 1)
    prev_week_close = np.roll(close_1w, 1)
    prev_week_high[0] = np.nan
    prev_week_low[0] = np.nan
    prev_week_close[0] = np.nan
    
    # Weekly pivot point
    pp = (prev_week_high + prev_week_low + prev_week_close) / 3
    # Weekly resistance and support levels (Standard pivot formula)
    r1 = 2 * pp - prev_week_low
    s1 = 2 * pp - prev_week_high
    r2 = pp + (prev_week_high - prev_week_low)
    s2 = pp - (prev_week_high - prev_week_low)
    
    # Align weekly pivot levels to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA(50) for trend filter
    close_1d_series = pd.Series(close_1d)
    ema_50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: volume > 1.8x average volume (20-period)
    vol_series = pd.Series(volume)
    avg_vol = vol_series.rolling(window=20, min_periods=20).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 50  # for 50-period EMA + weekly lookback
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(avg_vol[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        if position == 0:
            # Long: price breaks above weekly R2 AND above 1d EMA50 with volume confirmation
            if price > r2_aligned[i] and price > ema_50_1d_aligned[i] and vol > 1.8 * avg_vol[i]:
                position = 1
                signals[i] = position_size
            # Short: price breaks below weekly S2 AND below 1d EMA50 with volume confirmation
            elif price < s2_aligned[i] and price < ema_50_1d_aligned[i] and vol > 1.8 * avg_vol[i]:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below weekly S1
            if price < s1_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above weekly R1
            if price > r1_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1w_1d_Weekly_Pivot_EMA_Volume"
timeframe = "12h"
leverage = 1.0