#!/usr/bin/env python3
"""
1d_Camarilla_R3_S3_Breakout_1wTrend_Filter
Hypothesis: Daily Camarilla R3/S3 breakouts with 1-week EMA50 trend filter. 
Only trade breakouts in direction of weekly trend to avoid counter-trend whipsaws.
Uses volume confirmation to ensure breakout validity. Designed for low trade frequency 
(~10-20/year) with discrete position sizing (0.25) to minimize fee drag. 
Works in both bull and bear markets by aligning with higher-timeframe trend.
Camarilla levels provide precise intraday support/resistance derived from prior day range.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla levels and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1-week EMA50 for trend filter (using weekly data)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w, additional_delay_bars=1)
    
    # Calculate 20-period average volume for confirmation (on 1d)
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA50 weekly (50) + volume MA (20)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(vol_ma_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Get prior day's OHLC for Camarilla calculation (yesterday's completed 1d bar)
        # Since we're on 1d timeframe, prior day is i-1 in the 1d array
        idx_1d = i // 1  # 1d bars align 1:1 with 1d timeframe prices
        if idx_1d < 1:
            continue
            
        # Calculate Camarilla levels from prior day (idx_1d-1 in 1d array)
        prior_idx = idx_1d - 1
        if prior_idx < 0 or prior_idx >= len(high_1d):
            continue
            
        high_prior = high_1d[prior_idx]
        low_prior = low_1d[prior_idx]
        close_prior = close_1d[prior_idx]
        
        # Camarilla R3 and S3 levels
        range_prior = high_prior - low_prior
        camarilla_r3 = close_prior + range_prior * 1.1 / 4
        camarilla_s3 = close_prior - range_prior * 1.1 / 4
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        vol_confirm = volume[i] > 1.5 * vol_ma_aligned[i]
        
        if position == 0:
            # Look for breakout signals with trend and volume confirmation
            # Long: price breaks above Camarilla R3 in uptrend (close > EMA50 weekly) + volume
            # Short: price breaks below Camarilla S3 in downtrend (close < EMA50 weekly) + volume
            long_signal = (close[i] > camarilla_r3) and (close[i] > ema50_1w_aligned[i]) and vol_confirm
            short_signal = (close[i] < camarilla_s3) and (close[i] < ema50_1w_aligned[i]) and vol_confirm
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit when price moves back below Camarilla S3 (mean reversion) or trend fails
            exit_signal = (close[i] < camarilla_s3) or (close[i] < ema50_1w_aligned[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when price moves back above Camarilla R3 (mean reversion) or trend fails
            exit_signal = (close[i] > camarilla_r3) or (close[i] > ema50_1w_aligned[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Camarilla_R3_S3_Breakout_1wTrend_Filter"
timeframe = "1d"
leverage = 1.0