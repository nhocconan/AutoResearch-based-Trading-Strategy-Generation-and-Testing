#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Weekly Pivot Breakout with Daily Trend Filter and Volume Confirmation
# Weekly pivot levels (R4/S4) represent strong institutional support/resistance from the prior week.
# Breakout above weekly R4 or below weekly S4 with volume confirms institutional participation.
# Daily EMA50 ensures alignment with intermediate-term trend to avoid counter-trend trades.
# Designed for 12-37 trades/year on 6h to minimize fee drag while capturing strong trending moves.
# Works in bull markets via long R4 breakouts in uptrend and in bear markets via short S4 breakdowns in downtrend.

name = "6h_WeeklyPivot_R4S4_Breakout_DailyEMA50_Trend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for HTF trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate daily EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Get weekly data for Camarilla pivot calculation (using weekly high/low/close)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Weekly Camarilla pivot levels (R4 and S4 only for breakout)
    # Pivot = (High + Low + Close) / 3
    pivot_1w = (high_1w + low_1w + close_1w) / 3
    # Range = High - Low
    rng_1w = high_1w - low_1w
    # Weekly Camarilla levels
    # R4 = Close + Range * 1.5000 (strongest resistance)
    # S4 = Close - Range * 1.5000 (strongest support)
    r4_1w = close_1w + rng_1w * 1.5000
    s4_1w = close_1w - rng_1w * 1.5000
    
    # Align Weekly Camarilla levels to 6h timeframe (use previous week's levels)
    r4_aligned = align_htf_to_ltf(prices, df_1w, r4_1w)
    s4_aligned = align_htf_to_ltf(prices, df_1w, s4_1w)
    
    # Calculate volume spike filter (20-period volume EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ema_20 * 2.0)  # Volume at least 2x average for confirmation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_aligned[i]) or np.isnan(r4_aligned[i]) or 
            np.isnan(s4_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above weekly R4 AND daily uptrend AND volume spike
            if (close[i] > r4_aligned[i] and 
                close[i] > ema_50_aligned[i] and  # daily uptrend
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below weekly S4 AND daily downtrend AND volume spike
            elif (close[i] < s4_aligned[i] and 
                  close[i] < ema_50_aligned[i] and  # daily downtrend
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below weekly pivot OR daily trend turns down
            pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
            if close[i] < pivot_aligned[i] or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above weekly pivot OR daily trend turns up
            pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
            if close[i] > pivot_aligned[i] or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals