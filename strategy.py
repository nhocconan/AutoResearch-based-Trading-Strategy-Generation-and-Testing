#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume confirmation
# Camarilla pivot levels from 4h chart provide institutional support/resistance for breakouts/breakdowns.
# 4h EMA50 ensures alignment with 4h trend to avoid counter-trend trades. Volume spike confirms participation.
# Session filter (08-20 UTC) reduces noise. Designed for 60-150 total trades over 4 years (15-37/year) to minimize fee drag.
# Works in bull markets via breakout continuation and in bear markets via breakdown shorts with trend filter.

name = "1h_Camarilla_R3S3_4hEMA50_VolumeSpike"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Get 4h data for Camarilla pivot levels
    # Calculate Camarilla pivot levels (R3, S3) from previous 4h bar
    prev_4h_high = df_4h['high'].shift(1).values
    prev_4h_low = df_4h['low'].shift(1).values
    prev_4h_close = df_4h['close'].shift(1).values
    
    camarilla_r3 = prev_4h_close + 1.1 * (prev_4h_high - prev_4h_low) * 1.1 / 4
    camarilla_s3 = prev_4h_close - 1.1 * (prev_4h_high - prev_4h_low) * 1.1 / 4
    
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):  # Start from 1 to have previous 4h bar data
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: 20-period EMA on 1h
        if i >= 19:
            vol_ema_20 = pd.Series(volume[i-19:i+1]).ewm(span=20, adjust=False, min_periods=1).mean().iloc[-1]
        else:
            vol_ema_20 = volume[i]
        volume_spike = volume[i] > (1.5 * vol_ema_20)
        
        if position == 0:
            # Long: price breaks above Camarilla R3 in 4h uptrend with volume spike
            if close[i] > camarilla_r3_aligned[i] and ema_50_4h_aligned[i] < close[i] and volume_spike:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below Camarilla S3 in 4h downtrend with volume spike
            elif close[i] < camarilla_s3_aligned[i] and ema_50_4h_aligned[i] > close[i] and volume_spike:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price breaks below Camarilla S3 or loses 4h uptrend
            if close[i] < camarilla_s3_aligned[i] or ema_50_4h_aligned[i] >= close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price breaks above Camarilla R3 or loses 4h downtrend
            if close[i] > camarilla_r3_aligned[i] or ema_50_4h_aligned[i] <= close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals