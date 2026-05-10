#!/usr/bin/env python3
# 6h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike
# Hypothesis: Camarilla pivot levels from 1d: fade at R3/S3, breakout continuation at R4/S4.
# R3/S3 act as strong support/resistance in ranging markets; break of R4/S4 indicates momentum shift.
# Combined with 1d EMA trend filter and volume spike for confirmation. Works in bull/bear by aligning with daily trend.
# Targets 20-50 trades/year to minimize fee drag.

name = "6h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0

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
    
    # Get daily data for Camarilla calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    # R4 = Close + (High - Low) * 1.500
    # R3 = Close + (High - Low) * 1.250
    # S3 = Close - (High - Low) * 1.250
    # S4 = Close - (High - Low) * 1.500
    # We use previous day's values to avoid look-ahead
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Calculate Camarilla levels for each day
    high_low_diff = prev_high - prev_low
    r4 = prev_close + high_low_diff * 1.500
    r3 = prev_close + high_low_diff * 1.250
    s3 = prev_close - high_low_diff * 1.250
    s4 = prev_close - high_low_diff * 1.500
    
    # Align to 6h timeframe (wait for previous day's close)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Daily EMA trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation (24-period MA for 6h = 4 days)
    volume_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 24  # Warmup for volume MA
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(r4_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Daily trend filter
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        # Volume confirmation
        volume_confirm = volume[i] > volume_ma[i] * 2.0
        
        if position == 0:
            # Long breakout: price breaks above R4 with volume and uptrend
            if close[i] > r4_aligned[i] and volume_confirm and uptrend:
                signals[i] = 0.25
                position = 1
            # Short breakdown: price breaks below S4 with volume and downtrend
            elif close[i] < s4_aligned[i] and volume_confirm and downtrend:
                signals[i] = -0.25
                position = -1
            # Long fade: price approaches S3 in uptrend (mean reversion)
            elif close[i] <= s3_aligned[i] * 1.005 and volume_confirm and uptrend:
                signals[i] = 0.25
                position = 1
            # Short fade: price approaches R3 in downtrend (mean reversion)
            elif close[i] >= r3_aligned[i] * 0.995 and volume_confirm and downtrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price reaches R3 (take profit) or trend reversal
            if close[i] >= r3_aligned[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reaches S3 (take profit) or trend reversal
            if close[i] <= s3_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals