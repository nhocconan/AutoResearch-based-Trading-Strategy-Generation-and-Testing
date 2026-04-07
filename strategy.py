#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 1d Weekly Pivot Range Breakout with Volume Confirmation
# Hypothesis: Price breaking above/below weekly pivot levels on 1d with volume confirmation
# captures institutional interest in both bull and bear markets. Weekly pivots act as
# dynamic support/resistance, and volume confirms genuine breakouts. Uses 1w trend filter
# to avoid counter-trend trades. Target: 15-25 trades/year (60-100 total) for low fee drag.

name = "1d_weekly_pivot_range_breakout_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot points and trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using prior week's OHLC)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot: P = (H + L + C) / 3
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    # Support 1: S1 = 2*P - H
    s1_1w = 2 * pivot_1w - high_1w
    # Resistance 1: R1 = 2*P - L
    r1_1w = 2 * pivot_1w - low_1w
    
    # Align weekly levels to daily (using previous week's values)
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    
    # Weekly trend: EMA(20) on weekly close
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False).mean().values
    ema_trend_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Volume confirmation: volume > 1.5 * 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if required data not available
        if np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) \
           or np.isnan(ema_trend_aligned[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below pivot or trend changes
            if close[i] < pivot_aligned[i] or close[i] < ema_trend_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price crosses above pivot or trend changes
            if close[i] > pivot_aligned[i] or close[i] > ema_trend_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Long: price breaks above R1 with volume confirmation in uptrend
            if close[i] > ema_trend_aligned[i]:  # Uptrend filter
                if high[i] > r1_aligned[i] and volume[i] > 1.5 * vol_ma[i]:
                    position = 1
                    signals[i] = 0.25
            # Short: price breaks below S1 with volume confirmation in downtrend
            elif close[i] < ema_trend_aligned[i]:  # Downtrend filter
                if low[i] < s1_aligned[i] and volume[i] > 1.5 * vol_ma[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals