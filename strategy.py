#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy combining weekly pivot points with 1d ATR-based regime filter
# Long when price breaks above weekly R1 with 1d ATR expansion (volatility breakout)
# Short when price breaks below weekly S1 with 1d ATR expansion
# Exit when price returns to weekly pivot (PP) or ATR contracts below average
# Uses weekly pivot structure for key levels and 1d ATR to filter for genuine breakouts
# Works in bull markets (buying breakouts above R1) and bear markets (selling breakdowns below S1)
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe
# Weekly pivots provide significant support/resistance that price respects
# ATR expansion confirms breakout strength and reduces false signals in ranging markets
# Discrete sizing 0.25 minimizes fee churn while maintaining adequate position size

name = "6h_WeeklyPivot_ATRBreakout_Regime"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get weekly data ONCE before loop for pivot points
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    # Weekly pivot calculation uses prior week's OHLC
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    weekly_open = df_1w['open'].values
    
    # Calculate weekly pivot points (using prior week's data)
    # PP = (H + L + C) / 3
    # R1 = (2 * PP) - L
    # S1 = (2 * PP) - H
    pp = (weekly_high + weekly_low + weekly_close) / 3.0
    r1 = (2 * pp) - weekly_low
    s1 = (2 * pp) - weekly_high
    
    # Align weekly pivot points to 6h timeframe (wait for completed weekly bar)
    pp_aligned = align_htf_to_ltf(prices, df_1w, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    
    # Get 1d data ONCE before loop for ATR regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:  # Need for ATR calculation
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ATR(14) for volatility regime filter
    # True Range = max[(high-low), abs(high-previous_close), abs(low-previous_close)]
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # First period
    tr2[0] = high_1d[0] - close_1d[0]  # First period
    tr3[0] = low_1d[0] - close_1d[0]   # First period
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate ATR average for contraction filter
    atr_avg_1d = pd.Series(atr_1d).rolling(window=50, min_periods=50).mean().values
    
    # Align ATR indicators to 6h timeframe
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    atr_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_avg_1d)
    
    # Session filter: 08-20 UTC (avoid low-volume Asian session)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(atr_1d_aligned[i]) or np.isnan(atr_avg_1d_aligned[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above weekly R1 with ATR expansion (volatility breakout)
            if (close[i] > r1_aligned[i] and close[i-1] <= r1_aligned[i-1] and
                atr_1d_aligned[i] > atr_avg_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly S1 with ATR expansion
            elif (close[i] < s1_aligned[i] and close[i-1] >= s1_aligned[i-1] and
                  atr_1d_aligned[i] > atr_avg_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to weekly pivot OR ATR contracts below average
            if close[i] <= pp_aligned[i] or atr_1d_aligned[i] < atr_avg_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to weekly pivot OR ATR contracts below average
            if close[i] >= pp_aligned[i] or atr_1d_aligned[i] < atr_avg_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals