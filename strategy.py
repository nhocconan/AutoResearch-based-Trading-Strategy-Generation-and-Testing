#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume spike
# Uses 4h Camarilla levels for structure, 4h EMA50 for trend direction, and volume spike for confirmation.
# Enters long when price breaks above R3 with volume confirmation and 4h EMA50 uptrend.
# Enters short when price breaks below S3 with volume confirmation and 4h EMA50 downtrend.
# Designed for 60-150 total trades over 4 years (15-37/year) on 1h timeframe to minimize fee drag.
# Uses 4h for signal direction, 1h only for entry timing precision. Session filter (08-20 UTC) reduces noise.

name = "1h_Camarilla_R3S3_Breakout_4hEMA50_Trend_VolumeSpike"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) - prices.index is DatetimeIndex
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for Camarilla calculation - ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Camarilla levels for each 4h bar
    # R3 = close + 1.1*(high-low), S3 = close - 1.1*(high-low)
    camarilla_range = high_4h - low_4h
    r3_4h = close_4h + 1.1 * camarilla_range
    s3_4h = close_4h - 1.1 * camarilla_range
    
    # Align Camarilla levels to 1h timeframe (wait for completed 4h bar)
    r3_4h_aligned = align_htf_to_ltf(prices, df_4h, r3_4h)
    s3_4h_aligned = align_htf_to_ltf(prices, df_4h, s3_4h)
    
    # Get 4h data for EMA50 trend filter - ONCE before loop
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align EMA50 to 1h timeframe (wait for completed 4h bar)
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Calculate volume spike filter (20-period volume MA)
    vol_ma_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ma_20 * 2.0)  # Volume at least 2x average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN or outside session
        if (np.isnan(r3_4h_aligned[i]) or np.isnan(s3_4h_aligned[i]) or 
            np.isnan(ema50_4h_aligned[i]) or np.isnan(volume_spike[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above R3 AND volume spike AND 4h EMA50 uptrend
            if (close[i] > r3_4h_aligned[i] and 
                volume_spike[i] and 
                close[i] > ema50_4h_aligned[i]):
                signals[i] = 0.20
                position = 1
            # Short conditions: price breaks below S3 AND volume spike AND 4h EMA50 downtrend
            elif (close[i] < s3_4h_aligned[i] and 
                  volume_spike[i] and 
                  close[i] < ema50_4h_aligned[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price re-enters Camarilla range (between S3 and R3) OR trend reverses
            if (close[i] >= s3_4h_aligned[i] and close[i] <= r3_4h_aligned[i]) or close[i] < ema50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price re-enters Camarilla range OR trend reverses
            if (close[i] >= s3_4h_aligned[i] and close[i] <= r3_4h_aligned[i]) or close[i] > ema50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals