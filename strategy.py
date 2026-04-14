#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Weekly pivot point system with volume confirmation and trend filter
# Uses 1d timeframe with weekly pivots from prior week (Monday's data)
# Long when price > weekly R2 + above 1d EMA50 + volume > 1.5x avg
# Short when price < weekly S2 + below 1d EMA50 + volume > 1.5x avg
# Exit on opposite touch of weekly S1/R1
# Designed for fewer trades (<50/year) to avoid fee drag, works in bull/bear via trend filter

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for weekly pivot points (using Monday's data)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly pivot points using prior week's OHLC (Monday's data)
    # For daily data, weekly pivot = (prev_week_high + prev_week_low + prev_week_close) / 3
    # We'll use 5-day lookback for weekly data (approximation)
    prev_week_high = np.roll(high_1d, 5)
    prev_week_low = np.roll(low_1d, 5)
    prev_week_close = np.roll(close_1d, 5)
    prev_week_high[:5] = np.nan
    prev_week_low[:5] = np.nan
    prev_week_close[:5] = np.nan
    
    # Weekly pivot point
    pp = (prev_week_high + prev_week_low + prev_week_close) / 3
    # Weekly resistance and support levels
    r1 = 2 * pp - prev_week_low
    s1 = 2 * pp - prev_week_high
    r2 = pp + (prev_week_high - prev_week_low)
    s2 = pp - (prev_week_high - prev_week_low)
    r3 = prev_week_high + 2 * (pp - prev_week_low)
    s3 = prev_week_low - 2 * (prev_week_high - pp)
    
    # Align weekly pivot levels to 1d timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Calculate 1d EMA(50) for trend filter
    close_series = pd.Series(close)
    ema_50 = close_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume confirmation: volume > 1.5x average volume (20-period)
    vol_series = pd.Series(volume)
    avg_vol = vol_series.rolling(window=20, min_periods=20).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 50  # for 50-period EMA + 5-day weekly lookback
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or
            np.isnan(ema_50[i]) or np.isnan(avg_vol[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        if position == 0:
            # Long: price breaks above weekly R2 AND above EMA50 with volume confirmation
            if price > r2_aligned[i] and price > ema_50[i] and vol > 1.5 * avg_vol[i]:
                position = 1
                signals[i] = position_size
            # Short: price breaks below weekly S2 AND below EMA50 with volume confirmation
            elif price < s2_aligned[i] and price < ema_50[i] and vol > 1.5 * avg_vol[i]:
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

name = "1d_Weekly_Pivot_EMA_Volume"
timeframe = "1d"
leverage = 1.0