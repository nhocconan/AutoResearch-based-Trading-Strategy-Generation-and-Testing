#!/usr/bin/env python3
"""
6h Candlestick Strength + Daily Trend Filter
Hypothesis: Strong daily closes with high close-to-open ratio indicate institutional conviction.
Combined with 6h bullish/bearish candle strength (close near high/low) to enter in direction
of daily trend. Works in bull via continuation and bear via mean reversion at extremes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get daily data for trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Daily close-to-open ratio (body strength)
    body_size = np.abs(df_1d['close'].values - df_1d['open'].values)
    candle_range = df_1d['high'].values - df_1d['low'].values
    daily_body_ratio = np.where(candle_range > 0, body_size / candle_range, 0.0)
    # Daily trend: strong bullish/bearish close
    daily_bull_strong = (df_1d['close'].values > df_1d['open'].values) & (daily_body_ratio > 0.6)
    daily_bear_strong = (df_1d['close'].values < df_1d['open'].values) & (daily_body_ratio > 0.6)
    
    # Align to 6h timeframe
    daily_bull_strong_aligned = align_htf_to_ltf(prices, df_1d, daily_bull_strong.astype(float))
    daily_bear_strong_aligned = align_htf_to_ltf(prices, df_1d, daily_bear_strong.astype(float))
    
    # 6h candle strength: close near high/low
    body_size_6h = np.abs(close - open_)
    candle_range_6h = high - low
    close_to_high = np.where(candle_range_6h > 0, (high - close) / candle_range_6h, 0.0)
    close_to_low = np.where(candle_range_6h > 0, (close - low) / candle_range_6h, 0.0)
    # Bullish 6h candle: close near high (small upper shadow)
    bullish_6h = close_to_high < 0.2
    # Bearish 6h candle: close near low (small lower shadow)
    bearish_6h = close_to_low < 0.2
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Warmup
    
    for i in range(start_idx, n):
        if np.isnan(daily_bull_strong_aligned[i]) or np.isnan(daily_bear_strong_aligned[i]):
            signals[i] = 0.0
            continue
        
        daily_bull = daily_bull_strong_aligned[i] > 0.5
        daily_bear = daily_bear_strong_aligned[i] > 0.5
        bull_candle = bullish_6h[i]
        bear_candle = bearish_6h[i]
        
        if position == 0:
            # Enter long: daily bullish strength + 6h bullish candle
            if daily_bull and bull_candle:
                signals[i] = 0.25
                position = 1
            # Enter short: daily bearish strength + 6h bearish candle
            elif daily_bear and bear_candle:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: daily turns bearish or 6h candle loses strength
            if daily_bear or not bull_candle:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: daily turns bullish or 6h candle loses strength
            if daily_bull or not bear_candle:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Candlestick_Strength_DailyTrend"
timeframe = "6h"
leverage = 1.0