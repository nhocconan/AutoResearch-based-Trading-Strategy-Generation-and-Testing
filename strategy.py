#!/usr/bin/env python3
"""
1d_Camarilla_Pivot_Breakout_1wTrend_VolumeConfirmation
Hypothesis: On daily timeframe, use weekly Camarilla R3/S3 breakouts with weekly EMA50 trend filter and volume confirmation (>2.0x 20-period average). Target 7-25 trades/year by requiring multiple confirmations (breakout + trend + volume) to avoid overtrading. Works in both bull (breakouts with trend) and bear (mean reversion at extremes during range) markets via Camarilla levels and volume filter.
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
    
    # Get weekly data for Camarilla calculation and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:  # Need sufficient data for weekly calculations
        return np.zeros(n)
    
    # Calculate weekly OHLC for Camarilla pivot points
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Camarilla levels R3/S3 (based on previous weekly bar's range)
    # Camarilla R3 = close + 1.1*(high - low)/2
    # Camarilla S3 = close - 1.1*(high - low)/2
    prev_high_1w = np.roll(high_1w, 1)
    prev_low_1w = np.roll(low_1w, 1)
    prev_close_1w = np.roll(close_1w, 1)
    
    # Set first value to NaN (no previous bar)
    prev_high_1w[0] = np.nan
    prev_low_1w[0] = np.nan
    prev_close_1w[0] = np.nan
    
    camarilla_r3 = prev_close_1w + 1.1 * (prev_high_1w - prev_low_1w) / 2
    camarilla_s3 = prev_close_1w - 1.1 * (prev_high_1w - prev_low_1w) / 2
    
    # Calculate weekly EMA50 for trend filter
    close_1w_series = pd.Series(close_1w)
    ema_50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align Camarilla levels and EMA to daily timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s3)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: volume > 2.0x 20-period average
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume / np.maximum(volume_ma, 1e-10) > 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need weekly EMA50 + volume MA
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Weekly trend alignment
        trend_1w_uptrend = close[i] > ema_50_1w_aligned[i]
        trend_1w_downtrend = close[i] < ema_50_1w_aligned[i]
        
        if position == 0:
            # Long: price breaks above R3 + weekly uptrend + volume spike
            # Require confirmation: price outside bands for 2 consecutive bars
            long_breakout = (close[i] > camarilla_r3_aligned[i]) and (close[i-1] > camarilla_r3_aligned[i-1])
            long_signal = long_breakout and trend_1w_uptrend and volume_spike[i]
            
            # Short: price breaks below S3 + weekly downtrend + volume spike
            short_breakout = (close[i] < camarilla_s3_aligned[i]) and (close[i-1] < camarilla_s3_aligned[i-1])
            short_signal = short_breakout and trend_1w_downtrend and volume_spike[i]
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price touches S3 OR weekly trend turns down
            if (close[i] < camarilla_s3_aligned[i] or not trend_1w_uptrend):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price touches R3 OR weekly trend turns up
            if (close[i] > camarilla_r3_aligned[i] or not trend_1w_downtrend):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Camarilla_Pivot_Breakout_1wTrend_VolumeConfirmation"
timeframe = "1d"
leverage = 1.0