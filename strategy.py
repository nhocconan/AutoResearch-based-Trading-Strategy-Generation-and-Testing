#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using weekly pivot points (from 1w HTF) with 1d trend filter and volume confirmation.
# Long when price breaks above weekly R1, 1d EMA50 uptrend, and volume > 1.5x 20-bar avg.
# Short when price breaks below weekly S1, 1d EMA50 downtrend, and volume > 1.5x 20-bar avg.
# Exit on opposite weekly pivot level (S1 for long exit, R1 for short exit).
# Weekly pivots provide structure from higher timeframe, reducing noise in 6h.
# Combined with 1d trend filter and volume confirmation to avoid false breakouts.
# Timeframe: 6h, HTF: 1w for pivots, 1d for trend.

name = "6h_WeeklyPivot_R1S1_1dEMA50_Trend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for weekly pivot points
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate weekly pivot points from prior completed 1w bar
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot: P = (H + L + C) / 3
    # R1 = 2*P - L, S1 = 2*P - H
    weekly_p = (high_1w + low_1w + close_1w) / 3.0
    weekly_r1 = 2.0 * weekly_p - low_1w
    weekly_s1 = 2.0 * weekly_p - high_1w
    
    # Align weekly pivot levels to 6h timeframe (wait for 1w bar to close)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
    
    # Load 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for EMA50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if (np.isnan(weekly_r1_aligned[i]) or np.isnan(weekly_s1_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_r1 = weekly_r1_aligned[i]
        curr_s1 = weekly_s1_aligned[i]
        curr_ema_50_1d = ema_50_1d_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above weekly R1, uptrend (close > 1d EMA50), volume confirmation
            if (curr_close > curr_r1 and 
                curr_close > curr_ema_50_1d and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly S1, downtrend (close < 1d EMA50), volume confirmation
            elif (curr_close < curr_s1 and 
                  curr_close < curr_ema_50_1d and 
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # Exit condition: price touches or goes below weekly S1 (mean reversion to pivot)
            if curr_close <= curr_s1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit condition: price touches or goes above weekly R1 (mean reversion to pivot)
            if curr_close >= curr_r1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals