#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with 1d weekly pivot regime filter and volume confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for weekly pivot levels (R1/S1, R2/S2, R3/S3) and EMA50 trend filter.
- Entry: Long when price breaks above Donchian(20) high AND price > weekly pivot S1 AND price > 1d EMA50 AND volume > 1.5x 20-period average volume.
         Short when price breaks below Donchian(20) low AND price < weekly pivot R1 AND price < 1d EMA50 AND volume > 1.5x 20-period average volume.
- Exit: Opposite Donchian breakout OR price crosses 1d EMA50 in opposite direction.
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- Weekly pivot regime filters out counter-trend breakouts in ranging markets.
- Volume confirmation ensures breakouts have conviction.
- Works in bull markets (buy breakouts above S1 with uptrend) and bear markets (sell breakdowns below R1 with downtrend).
- Estimated trades: ~100 total over 4 years (~25/year) based on Donchian breakout frequency with filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def ema(values, period):
    """Calculate Exponential Moving Average."""
    return pd.Series(values).ewm(span=period, adjust=False, min_periods=period).mean().values

def rolling_mean(values, period):
    """Calculate Simple Moving Average."""
    return pd.Series(values).rolling(window=period, min_periods=period).mean().values

def calculate_weekly_pivots(df_1d):
    """Calculate weekly pivot points from daily OHLC.
    Using prior week's high, low, close to calculate current week's pivots.
    """
    # Shift by 1 to use prior week's data (avoiding look-ahead)
    weekly_high = df_1d['high'].rolling(window=5, min_periods=5).max().shift(1)
    weekly_low = df_1d['low'].rolling(window=5, min_periods=5).min().shift(1)
    weekly_close = df_1d['close'].rolling(window=5, min_periods=5).mean().shift(1)
    
    pivot = (weekly_high + weekly_low + weekly_close) / 3
    r1 = 2 * pivot - weekly_low
    s1 = 2 * pivot - weekly_high
    r2 = pivot + (weekly_high - weekly_low)
    s2 = pivot - (weekly_high - weekly_low)
    r3 = weekly_high + 2 * (pivot - weekly_low)
    s3 = weekly_low - 2 * (weekly_high - pivot)
    
    return pivot, r1, r2, r3, s1, s2, s3

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d trend filter: EMA50
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    ema50_1d = ema(df_1d['close'].values, 50)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate weekly pivots from 1d data
    pivot, r1, r2, r3, s1, s2, s3 = calculate_weekly_pivots(df_1d)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1.values)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1.values)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2.values)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2.values)
    
    # Donchian(20) on 6h
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.5 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 60  # Need sufficient data for Donchian/EMA/volume
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(volume_confirmed[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Exit conditions: opposite Donchian breakout OR price crosses 1d EMA50 in opposite direction
        if position != 0:
            # Exit long: price breaks below Donchian low OR price falls below 1d EMA50
            if position == 1:
                if curr_low < donchian_low[i] or curr_close < ema50_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price breaks above Donchian high OR price rises above 1d EMA50
            elif position == -1:
                if curr_high > donchian_high[i] or curr_close > ema50_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Donchian breakout with volume confirmation and pivot/EMA filters
        if position == 0:
            # Long: price breaks above Donchian high AND above weekly S1 AND above 1d EMA50 AND volume confirmed
            if (curr_high > donchian_high[i] and 
                curr_close > s1_aligned[i] and 
                curr_close > ema50_1d_aligned[i] and 
                volume_confirmed[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low AND below weekly R1 AND below 1d EMA50 AND volume confirmed
            elif (curr_low < donchian_low[i] and 
                  curr_close < r1_aligned[i] and 
                  curr_close < ema50_1d_aligned[i] and 
                  volume_confirmed[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_WeeklyPivotRegime_EMA50_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0