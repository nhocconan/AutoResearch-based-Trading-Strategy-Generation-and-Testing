#!/usr/bin/env python3
"""
6h_Weekly_Pivot_Reversal
Hypothesis: Price often reverses at weekly pivot levels (R1/S1) during low volatility periods.
We use weekly pivot points calculated from prior week's OHLC, confirmed by Bollinger Band squeeze
(indicating low volatility) and mean-reversion signals when price touches pivot levels.
This strategy works in both bull and bear markets by capturing mean-reversion at key levels
while avoiding trending periods via volatility filter. Target: 20-30 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot points
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) == 0:
        return np.zeros(n)
    
    # Calculate weekly pivot points: P = (H+L+C)/3, R1 = 2*P - L, S1 = 2*P - H
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    r1 = 2 * pivot - weekly_low
    s1 = 2 * pivot - weekly_high
    
    # Align weekly pivots to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_weekly, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_weekly, r1)
    s1_aligned = align_htf_to_ltf(prices, df_weekly, s1)
    
    # Bollinger Bands (20-period, 2 std) for volatility regime
    close_series = pd.Series(close)
    bb_middle = close_series.rolling(window=20, min_periods=20).mean().values
    bb_std = close_series.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    bb_width = (bb_upper - bb_lower) / bb_middle  # Normalized width
    
    # Bollinger Band squeeze: low volatility when width < 20th percentile
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=50, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5, raw=False
    ).values
    low_volatility = bb_width_percentile < 0.2  # Bottom 20% = squeeze
    
    # Mean reversion signal: price touching pivot levels with rejection
    # Use 2-period RSI for short-term exhaustion
    delta = close_series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = pd.Series(gain).rolling(window=2, min_periods=2).mean().values
    avg_loss = pd.Series(loss).rolling(window=2, min_periods=2).mean().values
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 0)
    rsi_2 = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(bb_width[i]) or
            np.isnan(rsi_2[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        pivot_val = pivot_aligned[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        low_vol = low_volatility[i]
        rsi_val = rsi_2[i]
        
        if position == 0:
            # Look for mean reversion at pivot levels during low volatility
            # Near S1 with oversold RSI = long
            if low_vol and price <= s1_val * 1.002 and rsi_val < 30:
                signals[i] = 0.25
                position = 1
            # Near R1 with overbought RSI = short
            elif low_vol and price >= r1_val * 0.998 and rsi_val > 70:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit when price reaches pivot or RSI normalizes
            if price >= pivot_val or rsi_val > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit when price reaches pivot or RSI normalizes
            if price <= pivot_val or rsi_val < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Weekly_Pivot_Reversal"
timeframe = "6h"
leverage = 1.0