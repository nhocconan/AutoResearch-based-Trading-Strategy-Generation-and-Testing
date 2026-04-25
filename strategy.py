#!/usr/bin/env python3
"""
6h_Camarilla_R3S3_Breakout_1dVolumeTrend_Confirm
Hypothesis: Camarilla R3/S3 breakouts on 6h with 1d volume trend confirmation (volume > 1.5x 20-bar MA) and price >/< 20-bar SMA for trend filter. 
Designed for 6h timeframe targeting 20-30 trades/year. Works in bull/bear by following price trend via SMA.
"""

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
    
    # Get 1d data for HTF volume trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d volume trend: volume > 1.5x 20-bar MA
    volume_1d = df_1d['volume'].values
    volume_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_trend_1d = volume_1d > (1.5 * volume_ma_1d)
    volume_trend_aligned = align_htf_to_ltf(prices, df_1d, volume_trend_1d, additional_delay_bars=1)
    
    # Calculate Camarilla levels from previous 1d bar
    # R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low), S3 = close - 1.1*(high-low), S4 = close - 1.5*(high-low)
    # But we need the previous completed 1d bar's OHLC
    prev_close_1d = df_1d['close'].shift(1).values
    prev_high_1d = df_1d['high'].shift(1).values
    prev_low_1d = df_1d['low'].shift(1).values
    prev_range_1d = prev_high_1d - prev_low_1d
    
    r3_1d = prev_close_1d + 1.1 * prev_range_1d
    s3_1d = prev_close_1d - 1.1 * prev_range_1d
    
    # Align Camarilla levels to 6h
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_1d, additional_delay_bars=1)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_1d, additional_delay_bars=1)
    
    # Calculate 20-bar SMA on 6h for trend filter
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for SMA and Camarilla
    start_idx = max(20, 30)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or
            np.isnan(sma_20[i]) or
            np.isnan(volume_trend_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:
            # Look for breakout signals with volume trend and price trend alignment
            long_signal = (close[i] > r3_aligned[i]) and volume_trend_aligned[i] and (close[i] > sma_20[i])
            short_signal = (close[i] < s3_aligned[i]) and volume_trend_aligned[i] and (close[i] < sma_20[i])
            
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
            # Exit when price breaks below S3 or volume trend fails
            exit_signal = (close[i] < s3_aligned[i]) or (~volume_trend_aligned[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when price breaks above R3 or volume trend fails
            exit_signal = (close[i] > r3_aligned[i]) or (~volume_trend_aligned[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Camarilla_R3S3_Breakout_1dVolumeTrend_Confirm"
timeframe = "6h"
leverage = 1.0