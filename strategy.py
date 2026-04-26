#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_12hEMA50_Trend_VolumeSpike_v1
Hypothesis: Camarilla pivot R1/S1 breakout on 4h with 12h EMA50 trend filter and volume confirmation (>1.5x 20-period MA).
Long when price breaks above R1 in 12h uptrend with volume spike. Short when price breaks below S1 in 12h downtrend with volume spike.
Uses discrete position sizing (0.25) to minimize fee churn. 
Camarilla levels derived from prior 1d OHLC. 
Designed to work in both bull and bear markets by following the 12h trend.
Target: 19-50 trades/year (75-200 total over 4 years).
Reduced trade frequency by tightening volume confirmation threshold from 1.3x to 1.5x and requiring EMA trend alignment for both entry and exit.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla calculation and 12h data for trend filter
    df_1d = get_htf_data(prices, '1d')
    df_12h = get_htf_data(prices, '12h')
    if len(df_1d) < 2 or len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from prior 1d OHLC
    # Using prior day's close to avoid look-ahead
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Shift by 1 to use prior day's OHLC for current day's levels
    close_1d_prev = np.roll(close_1d, 1)
    high_1d_prev = np.roll(high_1d, 1)
    low_1d_prev = np.roll(low_1d, 1)
    close_1d_prev[0] = np.nan
    high_1d_prev[0] = np.nan
    low_1d_prev[0] = np.nan
    
    # Camarilla R1, S1, R3, S3 levels
    # R1 = Close + (High - Low) * 1.1/12
    # S1 = Close - (High - Low) * 1.1/12
    # R3 = Close + (High - Low) * 1.1/4
    # S3 = Close - (High - Low) * 1.1/4
    camarilla_range = high_1d_prev - low_1d_prev
    r1 = close_1d_prev + camarilla_range * 1.1 / 12
    s1 = close_1d_prev - camarilla_range * 1.1 / 12
    r3 = close_1d_prev + camarilla_range * 1.1 / 4
    s3 = close_1d_prev - camarilla_range * 1.1 / 4
    
    # Align Camarilla levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # 12h EMA50 trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    uptrend_12h = close > ema_50_12h_aligned
    downtrend_12h = close < ema_50_12h_aligned
    
    # Volume confirmation: volume > 1.5x 20-period MA (tightened from 1.3x to reduce trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 50 for 12h EMA + 20 for volume MA + 1 for Camarilla shift)
    start_idx = 71
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:
            # Long: price breaks above R1 with 12h uptrend and volume spike
            if (close[i] > r1_aligned[i] and 
                uptrend_12h[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with 12h downtrend and volume spike
            elif (close[i] < s1_aligned[i] and 
                  downtrend_12h[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price closes below R3 (strong reversal) OR 12h trend changes to downtrend
            if (close[i] < r3_aligned[i] or not uptrend_12h[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price closes above S3 (strong reversal) OR 12h trend changes to uptrend
            if (close[i] > s3_aligned[i] or not downtrend_12h[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_12hEMA50_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0