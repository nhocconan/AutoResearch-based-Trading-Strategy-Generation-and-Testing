#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Weekly Pivot Reversal Strategy
# Uses weekly pivot points (PP, R1, R2, S1, S2) from the previous week to identify
# key support/resistance levels. Enters long near S1/S2 with bullish rejection
# (close > open and low touches pivot level) and short near R1/R2 with bearish
# rejection (close < open and high touches pivot level). Includes volume confirmation
# to avoid false breaks. Designed to work in both bull and bear markets by
# fading extreme weekly levels. Target: 50-150 total trades over 4 years (12-37/year).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data for pivot point calculation
    df_week = get_htf_data(prices, '1w')
    if len(df_week) < 5:
        return np.zeros(n)
    
    # Calculate weekly pivot points using prior week's OHLC
    # PP = (H + L + C) / 3
    # R1 = (2 * PP) - L
    # S1 = (2 * PP) - H
    # R2 = PP + (H - L)
    # S2 = PP - (H - L)
    high_week = df_week['high'].values
    low_week = df_week['low'].values
    close_week = df_week['close'].values
    
    pp = (high_week + low_week + close_week) / 3.0
    r1 = (2 * pp) - low_week
    s1 = (2 * pp) - high_week
    r2 = pp + (high_week - low_week)
    s2 = pp - (high_week - low_week)
    
    # Align weekly pivot points to 6h timeframe (using prior week's values)
    pp_aligned = align_htf_to_ltf(prices, df_week, pp)
    r1_aligned = align_htf_to_ltf(prices, df_week, r1)
    s1_aligned = align_htf_to_ltf(prices, df_week, s1)
    r2_aligned = align_htf_to_ltf(prices, df_week, r2)
    s2_aligned = align_htf_to_ltf(prices, df_week, s2)
    
    # Volume average (20-period on 6h) for confirmation
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size (25% of capital)
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or np.isnan(vol_avg[i])):
            continue
        
        # Long entry: price near S1 or S2 with bullish rejection and volume confirmation
        # Bullish rejection: close > open and low touches or goes below pivot level
        long_condition = False
        if (close[i] > open_price if 'open_price' in locals() else close[i] > close[i-1] if i > 0 else False):
            # More robust: check for bullish candle (close > open)
            bullish = close[i] > prices['open'].iloc[i]
            near_s1 = low[i] <= s1_aligned[i] * 1.001  # Allow 0.1% slippage
            near_s2 = low[i] <= s2_aligned[i] * 1.001
            vol_ok = volume[i] > vol_avg[i]
            if bullish and (near_s1 or near_s2) and vol_ok:
                long_condition = True
        
        # Short entry: price near R1 or R2 with bearish rejection and volume confirmation
        # Bearish rejection: close < open and high touches or goes above pivot level
        short_condition = False
        bearish = close[i] < prices['open'].iloc[i]
        near_r1 = high[i] >= r1_aligned[i] * 0.999  # Allow 0.1% slippage
        near_r2 = high[i] >= r2_aligned[i] * 0.999
        vol_ok = volume[i] > vol_avg[i]
        if bearish and (near_r1 or near_r2) and vol_ok:
            short_condition = True
        
        # Execute signals
        if long_condition and position <= 0:
            position = 1
            signals[i] = base_size
        elif short_condition and position >= 0:
            position = -1
            signals[i] = -base_size
        # Exit on opposite signal or when price moves back toward pivot
        elif position == 1 and (short_condition or close[i] > pp_aligned[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (long_condition or close[i] < pp_aligned[i]):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "6h_WeeklyPivot_Reversal"
timeframe = "6h"
leverage = 1.0