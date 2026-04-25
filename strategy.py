#!/usr/bin/env python3
"""
6h_Weekly_Pivot_Regime_Trend
Hypothesis: Use weekly pivot levels to define trend regime (bull/bear/range) and trade 6h breakouts in direction of weekly trend.
Weekly pivot provides robust long-term structure. In bull regime (price above weekly R1), only long breakouts above weekly R2.
In bear regime (price below weekly S1), only short breakouts below weekly S2.
In range regime (between S1 and R1), fade at weekly R1/S1 with mean reversion.
Volume spike confirmation reduces false signals. Discrete sizing 0.25 to manage risk and minimize fee churn.
Target: 12-30 trades/year to stay within fee drag limits for 6h timeframe.
"""

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
    
    # Get weekly data for pivot levels and trend regime
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate weekly pivot points (standard floor trader pivots)
    # Using previous week's OHLC
    prev_week_high = df_1w['high'].shift(1).values
    prev_week_low = df_1w['low'].shift(1).values
    prev_week_close = df_1w['close'].shift(1).values
    
    pivot = (prev_week_high + prev_week_low + prev_week_close) / 3.0
    r1 = 2 * pivot - prev_week_low
    s1 = 2 * pivot - prev_week_high
    r2 = pivot + (prev_week_high - prev_week_low)
    s2 = pivot - (prev_week_high - prev_week_low)
    r3 = prev_week_high + 2 * (pivot - prev_week_low)
    s3 = prev_week_low - 2 * (prev_week_high - pivot)
    
    # Align weekly pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    
    # Volume spike: current volume > 1.8x 24-period average (4 days on 6h)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (1.8 * vol_ma_24)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for weekly data (1 week shift) and volume MA (24)
    start_idx = max(24, 1)  # weekly data already shifted by 1 in calculation
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or
            np.isnan(vol_ma_24[i])):
            signals[i] = 0.0
            continue
        
        # Determine weekly regime based on price relative to weekly S1/R1
        weekly_bull = close[i] > r1_aligned[i]
        weekly_bear = close[i] < s1_aligned[i]
        weekly_range = (close[i] >= s1_aligned[i]) and (close[i] <= r1_aligned[i])
        
        if position == 0:
            # Regime-based entries
            if weekly_bull:
                # Bull regime: only look for longs above R2
                long_setup = (close[i] > r2_aligned[i]) and volume_spike[i]
                if long_setup:
                    signals[i] = 0.25
                    position = 1
            elif weekly_bear:
                # Bear regime: only look for shorts below S2
                short_setup = (close[i] < s2_aligned[i]) and volume_spike[i]
                if short_setup:
                    signals[i] = -0.25
                    position = -1
            else:  # weekly_range
                # Range regime: fade at weekly R1/S1
                long_setup = (close[i] < s1_aligned[i]) and volume_spike[i]  # mean reversion long from S1
                short_setup = (close[i] > r1_aligned[i]) and volume_spike[i]  # mean reversion short from R1
                if long_setup:
                    signals[i] = 0.25
                    position = 1
                elif short_setup:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: weekly regime turns bearish OR price reaches weekly R3 (take profit)
            if weekly_bear or (close[i] >= r3_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: weekly regime turns bullish OR price reaches weekly S3 (take profit)
            if weekly_bull or (close[i] <= s3_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Weekly_Pivot_Regime_Trend"
timeframe = "6h"
leverage = 1.0