#!/usr/bin/env python3
"""
1d_Camarilla_R3S3_Breakout_1wTrend_VolumeConfirm_v3
Hypothesis: Daily Camarilla R3/S3 breakouts with 1-week EMA50 trend filter and volume confirmation (1.5x 20-day avg). In trending markets, breakouts at extreme Camarilla levels (R3/S3) capture strong moves. Volume confirms breakout validity. Designed for 1d timeframe targeting 15-25 trades/year. Works in bull/bear by following the weekly trend direction.
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
    
    # Get 1w data for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate EMA50 on 1w data for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Align EMA50 to 1d timeframe (1-week lagged for completed bar)
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w, additional_delay_bars=1)
    
    # Trend: bullish when price > EMA50, bearish when price < EMA50
    trend_bullish = ema_50_aligned > 0  # Will be replaced with actual comparison
    trend_bearish = ema_50_aligned > 0  # Will be replaced with actual comparison
    
    # Get 1d data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Camarilla levels on 1d data (based on previous day's OHLC)
    # Camarilla: R3 = C + ((H-L) * 1.1/4), S3 = C - ((H-L) * 1.1/4)
    camarilla_r3_1d = close_1d + ((high_1d - low_1d) * 1.1 / 4)
    camarilla_s3_1d = close_1d - ((high_1d - low_1d) * 1.1 / 4)
    
    # Align Camarilla levels to 1d timeframe (1-day lagged for completed bar)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3_1d, additional_delay_bars=1)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3_1d, additional_delay_bars=1)
    camarilla_c_1d = close_1d  # Camarilla C is close
    camarilla_c_aligned = align_htf_to_ltf(prices, df_1d, camarilla_c_1d, additional_delay_bars=1)
    
    # Volume confirmation: 1.5x 20-day average volume
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA50 and Camarilla
    start_idx = max(50, 30)  # 50 for EMA50 warmup
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(camarilla_c_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine trend direction
        trend_bullish = close[i] > ema_50_aligned[i]
        trend_bearish = close[i] < ema_50_aligned[i]
        
        if position == 0:
            # Look for breakout signals in direction of weekly trend with volume confirmation
            long_signal = (close[i] > camarilla_r3_aligned[i]) and trend_bullish and volume_spike[i]
            short_signal = (close[i] < camarilla_s3_aligned[i]) and trend_bearish and volume_spike[i]
            
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
            # Exit when price moves back below Camarilla C (mean reversion to midpoint)
            exit_signal = close[i] < camarilla_c_aligned[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when price moves back above Camarilla C (mean reversion to midpoint)
            exit_signal = close[i] > camarilla_c_aligned[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Camarilla_R3S3_Breakout_1wTrend_VolumeConfirm_v3"
timeframe = "1d"
leverage = 1.0