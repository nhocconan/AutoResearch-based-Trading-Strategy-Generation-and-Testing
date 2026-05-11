#!/usr/bin/env python3
"""
6H_WeeklyPivot_DailyTrend_Continuation
Hypothesis: Weekly pivot levels define institutional support/resistance. Daily trend filters direction.
In bull markets, buy pullbacks to weekly S1/S2 with daily uptrend. In bear markets, sell rallies to weekly R1/R2 with daily downtrend.
6h timeframe reduces trade frequency vs lower timeframes, minimizing fee drag in ranging 2025 market.
Target: 12-37 trades/year (50-150 total over 4 years).
"""

name = "6H_WeeklyPivot_DailyTrend_Continuation"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE for pivot levels
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points (standard formula)
    pivot = (high_1w + low_1w + close_1w) / 3.0
    r1 = 2 * pivot - low_1w
    s1 = 2 * pivot - high_1w
    r2 = pivot + (high_1w - low_1w)
    s2 = pivot - (high_1w - low_1w)
    
    # Align weekly pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    
    # Load daily data ONCE for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume filter: 24-period EMA for spike detection (4 periods per day on 6h)
    vol_ema24 = pd.Series(volume).ewm(span=24, min_periods=24, adjust=False).mean().values
    volume_ok = volume > vol_ema24 * 1.5
    
    # Fixed position size to minimize churn
    position_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or np.isnan(ema34_1d_aligned[i]) or
            np.isnan(volume_ok[i])):
            if position == 1:
                signals[i] = 0.0
            elif position == -1:
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        # Conditions
        price_above_ema1d = close[i] > ema34_1d_aligned[i]
        price_below_ema1d = close[i] < ema34_1d_aligned[i]
        
        if position == 0:
            # Long: Pullback to weekly support in daily uptrend
            if (close[i] <= s1_aligned[i] * 1.02 and close[i] >= s2_aligned[i] * 0.98 and
                price_above_ema1d and volume_ok[i]):
                signals[i] = position_size
                position = 1
            # Short: Rally to weekly resistance in daily downtrend
            elif (close[i] >= r1_aligned[i] * 0.98 and close[i] <= r2_aligned[i] * 1.02 and
                  price_below_ema1d and volume_ok[i]):
                signals[i] = -position_size
                position = -1
        else:
            # Exit conditions - trend reversal or opposite pivot touch
            if position == 1:
                # Exit: Daily trend turns down OR price reaches weekly resistance
                if (close[i] < ema34_1d_aligned[i] or 
                    close[i] >= r1_aligned[i] * 0.995):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = position_size
            elif position == -1:
                # Exit: Daily trend turns up OR price reaches weekly support
                if (close[i] > ema34_1d_aligned[i] or 
                    close[i] <= s1_aligned[i] * 1.005):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -position_size
    
    return signals