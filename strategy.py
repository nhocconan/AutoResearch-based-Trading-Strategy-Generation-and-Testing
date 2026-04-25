#!/usr/bin/env python3
"""
4h_Camarilla_R3S3_Breakout_12hTrend_VolumeSpike
Hypothesis: Camarilla R3/S3 breakouts on 4h with 12h EMA20 trend filter and volume spike confirmation.
Uses 12h EMA20 for trend direction (more responsive than 1d EMA50) to capture both bull and bear trends.
Volume spike (>2x 20-bar average) confirms breakout strength. Exits on reversion to Camarilla C (midpoint).
Discrete position sizing (0.25) minimizes fee churn. Target: 20-50 trades/year.
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
    
    # Get 12h data for HTF trend filter and Camarilla levels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate EMA20 on 12h close for trend filter
    ema20_12h = pd.Series(close_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate Camarilla levels on 12h data (based on previous bar's OHLC)
    camarilla_r3_12h = close_12h + ((high_12h - low_12h) * 1.1 / 4)
    camarilla_s3_12h = close_12h - ((high_12h - low_12h) * 1.1 / 4)
    camarilla_c_12h = close_12h  # Camarilla C is the close
    
    # Align HTF indicators to 4h timeframe (completed 12h bar lag)
    ema20_aligned = align_htf_to_ltf(prices, df_12h, ema20_12h, additional_delay_bars=1)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r3_12h, additional_delay_bars=1)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s3_12h, additional_delay_bars=1)
    camarilla_c_aligned = align_htf_to_ltf(prices, df_12h, camarilla_c_12h, additional_delay_bars=1)
    
    # Volume confirmation: 2.0x 20-bar average volume
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA20
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema20_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(camarilla_c_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:
            # Look for breakout signals in direction of 12h trend with volume confirmation
            # Long: price breaks above R3 in uptrend (close > EMA20)
            # Short: price breaks below S3 in downtrend (close < EMA20)
            long_signal = (close[i] > camarilla_r3_aligned[i]) and (close[i] > ema20_aligned[i]) and volume_spike[i]
            short_signal = (close[i] < camarilla_s3_aligned[i]) and (close[i] < ema20_aligned[i]) and volume_spike[i]
            
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

name = "4h_Camarilla_R3S3_Breakout_12hTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0