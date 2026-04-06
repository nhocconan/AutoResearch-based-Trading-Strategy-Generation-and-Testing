#!/usr/bin/env python3
"""
6h Williams %R Reversal with 1d Trend Filter
Hypothesis: Williams %R identifies overbought/oversold conditions for mean reversion entries.
In ranging markets, reversals at extremes work well. In trending markets, we filter by 1d EMA50
to only take reversals in the direction of the higher timeframe trend. Works in both bull and bear
markets by adapting to the 1d trend. Target: 80-150 total trades over 4 years (20-38/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_williamsr_1d_trend_filter_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Williams %R (14-period)
    williams_r = np.full(n, np.nan)
    if n >= 14:
        for i in range(14, n):
            highest_high = np.max(high[i-13:i+1])
            lowest_low = np.min(low[i-13:i+1])
            if highest_high != lowest_low:
                williams_r[i] = -100 * (highest_high - close[i]) / (highest_high - lowest_low)
            else:
                williams_r[i] = -50  # neutral when no range
    
    # Get 1d data for trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # EMA50 on 1d close
    ema_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema_1d[49] = np.mean(close_1d[:50])
        for i in range(50, len(close_1d)):
            ema_1d[i] = (close_1d[i] * 2 + ema_1d[i-1] * 48) / 50
    
    # 1d trend: above EMA50 = bullish, below = bearish
    trend_1d = np.where(close_1d > ema_1d, 1, -1)
    
    # Align 1d trend to 6h timeframe
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_exit = 0
    
    # Start from warmup period
    start = 50  # Need enough data for Williams %R and EMA
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(williams_r[i]) or np.isnan(trend_1d_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            bars_since_exit += 1
            continue
        
        # Check exits
        if position == 1:  # long position
            # Exit: Williams %R rises above -20 (overbought) OR against 1d trend
            if williams_r[i] > -20 or trend_1d_aligned[i] == -1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
            bars_since_exit += 1
        elif position == -1:  # short position
            # Exit: Williams %R falls below -80 (oversold) OR against 1d trend
            if williams_r[i] < -80 or trend_1d_aligned[i] == 1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
            bars_since_exit += 1
        else:
            # Look for entries with minimum bars since exit
            if bars_since_exit >= 6:  # minimum 6 bars (1 day) between trades
                # Long: Williams %R crosses below -80 (oversold) with bullish 1d trend
                if i > 1 and williams_r[i] <= -80 and williams_r[i-1] > -80 and trend_1d_aligned[i] == 1:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                    bars_since_exit = 0
                # Short: Williams %R crosses above -20 (overbought) with bearish 1d trend
                elif i > 1 and williams_r[i] >= -20 and williams_r[i-1] < -20 and trend_1d_aligned[i] == -1:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
                    bars_since_exit = 0
                else:
                    signals[i] = 0.0
                    bars_since_exit += 1
            else:
                signals[i] = 0.0
                bars_since_exit += 1
    
    return signals