#!/usr/bin/env python3
# 12h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike
# Hypothesis: Camarilla pivot levels (R3/S3) act as strong support/resistance on daily charts.
# Breakouts above R3 or below S3 with volume confirmation and daily trend alignment capture
# significant moves. Works in bull markets (breakouts above R3 in uptrend) and bear markets
# (breakdowns below S3 in downtrend). Uses volume spike to avoid false breakouts and
# maintains low trade frequency for 12h timeframe.
# Entry: Long when price breaks above R3(1d) with volume spike and daily uptrend.
#        Short when price breaks below S3(1d) with volume spike and daily downtrend.
# Exit: Reverse signal or trend breakdown.

name = "12h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike"
timeframe = "12h"
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
    
    # Get daily data for Camarilla calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels (R3, S3) from previous day
    # R3 = close + 1.1 * (high - low)
    # S3 = close - 1.1 * (high - low)
    # Using previous day's data to avoid look-ahead
    daily_close = df_1d['close'].values
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    
    # Shift by 1 to use previous day's data
    prev_close = np.roll(daily_close, 1)
    prev_high = np.roll(daily_high, 1)
    prev_low = np.roll(daily_low, 1)
    prev_close[0] = np.nan  # First day has no previous
    
    # Calculate Camarilla levels
    camarilla_r3 = prev_close + 1.1 * (prev_high - prev_low)
    camarilla_s3 = prev_close - 1.1 * (prev_high - prev_low)
    
    # Align to 12h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Daily trend filter: EMA50
    ema_50_1d = pd.Series(daily_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: 2-period volume average (24h = 2*12h)
    volume_ma = pd.Series(volume).rolling(window=2, min_periods=2).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need enough data for calculations
    start_idx = 1  # Need previous day data
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Daily trend filter
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Volume confirmation (volume spike: current > 2x average)
        volume_spike = volume[i] > volume_ma[i] * 2.0
        
        if position == 0:
            # Long entry: price breaks above R3 with volume spike and daily uptrend
            if (close[i] > camarilla_r3_aligned[i]) and volume_spike and uptrend:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below S3 with volume spike and daily downtrend
            elif (close[i] < camarilla_s3_aligned[i]) and volume_spike and downtrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below S3 or trend breaks
            if (close[i] < camarilla_s3_aligned[i]) or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above R3 or trend breaks
            if (close[i] > camarilla_r3_aligned[i]) or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals