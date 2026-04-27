#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily strategy using weekly pivot points (R3/S3 levels) with volume confirmation and weekly EMA(34) trend filter.
# Enters long when price breaks above S3 with volume, short when breaks below R3 with volume.
# Designed for ~10-25 trades/year by requiring significant breakouts (R3/S3) rather than minor S1/R1 levels.
# Works in bull/bear: buys support breaks, sells resistance breaks.
# Uses strict volume filter (volume > 2.5x 30-period average) to avoid false breakouts.
# Exit when price returns to weekly pivot or trend changes.
# Timeframe: 1d, HTF: 1w

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot calculation and trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points (using prior week OHLC)
    high_prev = np.roll(high_1w, 1)
    low_prev = np.roll(low_1w, 1)
    close_prev = np.roll(close_1w, 1)
    high_prev[0] = np.nan
    low_prev[0] = np.nan
    close_prev[0] = np.nan
    
    pivot = (high_prev + low_prev + close_prev) / 3.0
    r1 = 2 * pivot - low_prev
    s1 = 2 * pivot - high_prev
    r2 = pivot + (high_prev - low_prev)
    s2 = pivot - (high_prev - low_prev)
    r3 = high_prev + 2 * (pivot - low_prev)
    s3 = low_prev - 2 * (high_prev - pivot)
    
    # Align weekly pivots to daily
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    
    # Weekly trend: price above/below weekly EMA(34)
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume filter: volume > 2.5 x 30-period average (daily) for significance
    vol_ma_30 = np.full(n, np.nan)
    for i in range(29, n):
        vol_ma_30[i] = np.mean(volume[i-29:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need pivots (1), weekly EMA (34), volume MA (30)
    start_idx = max(1, 34, 30)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(pivot_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ma_30[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_30[i]
        
        # Volume filter (strict)
        vol_filter = vol_now > 2.5 * vol_avg
        
        # Trend filters
        weekly_bullish = price > ema_34_1w_aligned[i]
        weekly_bearish = price < ema_34_1w_aligned[i]
        
        if position == 0:
            # Long: price breaks above S3 with volume and weekly bullish
            if price > s3_aligned[i] and vol_filter and weekly_bullish:
                signals[i] = size
                position = 1
            # Short: price breaks below R3 with volume and weekly bearish
            elif price < r3_aligned[i] and vol_filter and weekly_bearish:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns below pivot or weekly trend turns bearish
            if price < pivot_aligned[i] or not weekly_bullish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns above pivot or weekly trend turns bullish
            if price > pivot_aligned[i] or not weekly_bearish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_Pivot_S3R3_Volume_WeeklyTrend"
timeframe = "1d"
leverage = 1.0