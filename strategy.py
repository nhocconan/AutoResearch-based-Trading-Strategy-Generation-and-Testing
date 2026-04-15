#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Weekly Pivot Points with Volume Confirmation and Trend Filter
# Uses weekly pivot levels (support/resistance) as entry zones. Long when price bounces off S1/S2 with volume,
# short when price rejects R1/R2 with volume. Trend filter uses weekly EMA50 to avoid counter-trend trades.
# Works in bull markets (bounce off support) and bear markets (reject resistance). Target: 30-100 total trades.
# Timeframe: 1d, HTF: 1w

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data for pivot points and EMA50
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points (using prior week's OHLC)
    # P = (H + L + C) / 3
    # S1 = 2*P - H, S2 = P - (H - L)
    # R1 = 2*P - L, R2 = P + (H - L)
    pivots = (high_1w + low_1w + close_1w) / 3
    s1 = 2 * pivots - high_1w
    s2 = pivots - (high_1w - low_1w)
    r1 = 2 * pivots - low_1w
    r2 = pivots + (high_1w - low_1w)
    
    # Align pivot levels to daily timeframe (previous week's levels)
    # Shift by 1 to avoid look-ahead (use prior week's pivots)
    pivots_prev = np.roll(pivots, 1)
    s1_prev = np.roll(s1, 1)
    s2_prev = np.roll(s2, 1)
    r1_prev = np.roll(r1, 1)
    r2_prev = np.roll(r2, 1)
    pivots_prev[0] = np.nan
    s1_prev[0] = np.nan
    s2_prev[0] = np.nan
    r1_prev[0] = np.nan
    r2_prev[0] = np.nan
    
    pivots_aligned = align_htf_to_ltf(prices, df_1w, pivots_prev)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1_prev)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2_prev)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1_prev)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2_prev)
    
    # Weekly EMA50 for trend filter
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size (25% of capital)
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(s1_aligned[i]) or np.isnan(s2_aligned[i]) or
            np.isnan(r1_aligned[i]) or np.isnan(r2_aligned[i]) or
            np.isnan(ema_50_aligned[i])):
            continue
        
        # Volume condition: current volume > 1.5x 20-day median volume
        vol_median = np.median(volume[max(0, i-20):i+1])
        vol_ok = volume[i] > 1.5 * vol_median
        
        # Long: price near support (S1 or S2) with volume, in uptrend (price > weekly EMA50)
        near_support = (abs(close[i] - s1_aligned[i]) < 0.005 * close[i]) or \
                       (abs(close[i] - s2_aligned[i]) < 0.005 * close[i])
        if near_support and vol_ok and close[i] > ema_50_aligned[i] and position <= 0:
            position = 1
            signals[i] = base_size
        
        # Short: price near resistance (R1 or R2) with volume, in downtrend (price < weekly EMA50)
        near_resistance = (abs(close[i] - r1_aligned[i]) < 0.005 * close[i]) or \
                          (abs(close[i] - r2_aligned[i]) < 0.005 * close[i])
        if near_resistance and vol_ok and close[i] < ema_50_aligned[i] and position >= 0:
            position = -1
            signals[i] = -base_size
        
        # Exit: opposite signal or loss of volume confirmation
        if position == 1 and (near_resistance or not vol_ok):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (near_support or not vol_ok):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "1d_WeeklyPivot_Volume_TrendFilter"
timeframe = "1d"
leverage = 1.0