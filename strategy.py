#!/usr/bin/env python3
"""
12h_1d_camarilla_range_breakout
Strategy: 12h breakout with 1-day range filter
Timeframe: 12h
Leverage: 1.0
Hypothesis: Uses 1-day range compression (current range < 50% of 20-day average) to filter 12h Camarilla breakouts. Designed for low trade frequency (<25/year) to minimize fee drag while capturing momentum in both bull and bear markets via range contraction/expansion cycles. Target: 15-20 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_camarilla_range_breakout"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_price = prices['open'].values
    
    # Load higher timeframe data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 12h ATR for context
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_12h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 12h volume filter: volume > 1.5x 10-period average
    vol_ma_10 = pd.Series(volume).rolling(window=10, min_periods=10).mean().values
    
    # === 1-day Range compression (filter: current range < 50% of 20-day average) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1-day range
    range_1d = high_1d - low_1d
    # 20-day average range
    range_ma_20 = pd.Series(range_1d).rolling(window=20, min_periods=20).mean().values
    range_ratio = range_1d / range_ma_20
    range_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, range_ratio)
    
    # === 1-day Close (prior close for context) ===
    close_1d_shifted = np.roll(close_1d, 1)
    close_1d_shifted[0] = np.nan
    close_1d_prior = align_htf_to_ltf(prices, df_1d, close_1d_shifted)
    
    # === 1-day Camarilla (entry levels from prior 1-day) ===
    high_1d_shift = np.roll(high_1d, 1)
    low_1d_shift = np.roll(low_1d, 1)
    close_1d_shift = np.roll(close_1d, 1)
    high_1d_shift[0] = np.nan
    low_1d_shift[0] = np.nan
    close_1d_shift[0] = np.nan
    
    pivot_1d = (high_1d_shift + low_1d_shift + close_1d_shift) / 3
    range_1d_calc = high_1d_shift - low_1d_shift
    r3_1d = close_1d_shift + range_1d_calc * 1.166
    s3_1d = close_1d_shift - range_1d_calc * 1.166
    
    # Align 1-day Camarilla to 12h timeframe
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    
    # Session filter: 08-20 UTC (major sessions)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid or outside session
        if (np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or
            np.isnan(close_1d_prior[i]) or np.isnan(range_ratio_1d_aligned[i]) or np.isnan(vol_ma_10[i]) or
            not in_session[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price_close = close[i]
        price_open = open_price[i]
        volume_current = volume[i]
        vol_ma = vol_ma_10[i]
        
        # Volume confirmation: 12h volume must be expanded (1.5x 10-period average)
        volume_expanded = volume_current > 1.5 * vol_ma
        
        # Range compression: current 1-day range < 50% of 20-day average
        range_compressed = range_ratio_1d_aligned[i] < 0.5
        
        # Strong candle: close > open for longs, close < open for shorts
        strong_bullish = price_close > price_open
        strong_bearish = price_close < price_open
        
        # Long conditions: 12h closes above prior 1-day's R3 with volume expansion + range compression + strong bullish candle
        long_signal = volume_expanded and range_compressed and strong_bullish and (price_close > r3_1d_aligned[i])
        
        # Short conditions: 12h closes below prior 1-day's S3 with volume expansion + range compression + strong bearish candle
        short_signal = volume_expanded and range_compressed and strong_bearish and (price_close < s3_1d_aligned[i])
        
        # Exit when price returns to the 1-day pivot (mean reversion within prior 1-day's range)
        exit_long = position == 1 and price_close < pivot_1d_aligned[i]
        exit_short = position == -1 and price_close > pivot_1d_aligned[i]
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Hypothesis: Uses 1-day range compression (current range < 50% of 20-day average) to filter 12h Camarilla breakouts. Designed for low trade frequency (<25/year) to minimize fee drag while capturing momentum in both bull and bear markets via range contraction/expansion cycles. Target: 15-20 trades/year.