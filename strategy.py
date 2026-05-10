#!/usr/bin/env python3
"""
6h_RangeBreakout_WeeklyTrend_Volume
Hypothesis: Combine weekly trend filter with 6h range breakout and volume confirmation to capture momentum in both bull and bear markets.
Weekly EMA200 establishes long-term trend direction. 6h price breaking above/below prior 24-bar high/low with volume confirms momentum.
Designed for 12-30 trades/year on 6h timeframe with strict entry criteria to minimize fee drag.
"""

name = "6h_RangeBreakout_WeeklyTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get weekly data for trend filter
    df_weekly = get_htf_data(prices, '1w')
    
    if len(df_weekly) < 50:
        return np.zeros(n)
    
    # Weekly EMA200 for trend filter
    ema_200 = pd.Series(df_weekly['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_aligned = align_htf_to_ltf(prices, df_weekly, ema_200)
    
    # Get 6h price and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume filter: current volume > 1.8x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_filter = volume > vol_ema20 * 1.8
    
    # 6h range: highest high and lowest low of prior 24 bars (4 days)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    rolling_high = high_series.rolling(window=24, min_periods=24).max().shift(1).values
    rolling_low = low_series.rolling(window=24, min_periods=24).min().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need weekly EMA200 (200 bars) and 6h range (24 bars)
    start_idx = 200
    
    for i in range(start_idx, n):
        # Skip if weekly trend is not available
        if np.isnan(ema_200_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: above weekly EMA200 (bullish trend) AND price breaks above 24-bar high with volume
            if close[i] > ema_200_aligned[i] and high[i] > rolling_high[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: below weekly EMA200 (bearish trend) AND price breaks below 24-bar low with volume
            elif close[i] < ema_200_aligned[i] and low[i] < rolling_low[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below 24-bar low OR trend turns bearish
            if low[i] < rolling_low[i] or close[i] < ema_200_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above 24-bar high OR trend turns bullish
            if high[i] > rolling_high[i] or close[i] > ema_200_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals