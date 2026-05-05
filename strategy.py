#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using weekly pivot points (standard calculation, not Camarilla) for breakout confirmation
# Long when price breaks above weekly R2 AND price > 6h EMA50 AND volume > 1.5 * avg_volume(20) on 6h
# Short when price breaks below weekly S2 AND price < 6h EMA50 AND volume > 1.5 * avg_volume(20) on 6h
# Exit when price crosses back below/above weekly pivot point
# Uses discrete sizing 0.25 to balance return and risk
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe
# Weekly pivot points provide robust support/resistance from higher timeframe
# 6h EMA50 filters primary trend to avoid counter-trend trades
# Volume confirmation reduces false breakouts
# Works in bull markets (breakouts with uptrend) and bear markets (breakdowns with downtrend)

name = "6h_WeeklyPivot_R2S2_Breakout_6hEMA50_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data ONCE before loop for pivot points
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:  # Need at least one completed weekly bar
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points (standard calculation)
    # Pivot Point (PP) = (High + Low + Close) / 3
    # R1 = 2*PP - Low, S1 = 2*PP - High
    # R2 = PP + (High - Low), S2 = PP - (High - Low)
    pp_1w = (high_1w + low_1w + close_1w) / 3.0
    r2_1w = pp_1w + (high_1w - low_1w)
    s2_1w = pp_1w - (high_1w - low_1w)
    pivot_1w = pp_1w  # PP level for exit
    
    # Align weekly pivot levels to 6h timeframe (wait for completed weekly bar)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2_1w)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2_1w)
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    
    # Get 6h data ONCE before loop for EMA50 trend filter
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 50:  # Need enough for EMA50
        return np.zeros(n)
    close_6h = df_6h['close'].values
    
    # Calculate 6h EMA50
    close_6h_series = pd.Series(close_6h)
    ema50_6h = close_6h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_6h_aligned = align_htf_to_ltf(prices, df_6h, ema50_6h)
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume on 6h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or 
            np.isnan(pivot_aligned[i]) or np.isnan(ema50_6h_aligned[i]) or 
            np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above weekly R2, above 6h EMA50, volume confirmation, in session
            if close[i] > r2_aligned[i] and close[i] > ema50_6h_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below weekly S2, below 6h EMA50, volume confirmation, in session
            elif close[i] < s2_aligned[i] and close[i] < ema50_6h_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price crosses below weekly pivot point
            if close[i] < pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price crosses above weekly pivot point
            if close[i] > pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals