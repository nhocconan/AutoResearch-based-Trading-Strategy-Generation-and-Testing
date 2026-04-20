#!/usr/bin/env python3
# 6h_WeeklyPivot_Breakout_Volume_TrendFilter
# Hypothesis: Weekly pivot levels (PP, R1, S1) derived from 1w OHLC provide major institutional support/resistance.
# Breakout above R1 or below S1 with volume confirmation and 1d trend alignment (price above/below 1d EMA34) signals strong momentum.
# Works in bull markets (R1 breakouts with uptrend) and bear markets (S1 breakdowns with downtrend).
# Volume filter reduces false breakouts. Trend filter ensures trades align with higher-timeframe momentum.
# Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.

name = "6h_WeeklyPivot_Breakout_Volume_TrendFilter"
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
    
    # Get 1w data for weekly pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate weekly pivot points from previous week's OHLC
    # PP = (H + L + C) / 3
    # R1 = (2 * PP) - L
    # S1 = (2 * PP) - H
    prev_week_high = np.roll(df_1w['high'].values, 1)
    prev_week_low = np.roll(df_1w['low'].values, 1)
    prev_week_close = np.roll(df_1w['close'].values, 1)
    
    # First value will be invalid (no previous week), handled by alignment
    weekly_pp = (prev_week_high + prev_week_low + prev_week_close) / 3.0
    weekly_r1 = (2 * weekly_pp) - prev_week_low
    weekly_s1 = (2 * weekly_pp) - prev_week_high
    
    # Align weekly pivot levels to 6h timeframe
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
    
    # 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: volume > 1.8 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(35, 20)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(weekly_r1_aligned[i]) or np.isnan(weekly_s1_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above weekly R1 with volume confirmation and uptrend (price > 1d EMA34)
            if (close[i] > weekly_r1_aligned[i] and volume_confirm[i] and 
                close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below weekly S1 with volume confirmation and downtrend (price < 1d EMA34)
            elif (close[i] < weekly_s1_aligned[i] and volume_confirm[i] and 
                  close[i] < ema_34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below weekly S1 (reversal) or trend turns down
            if close[i] < weekly_s1_aligned[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above weekly R1 (reversal) or trend turns up
            if close[i] > weekly_r1_aligned[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals