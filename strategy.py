#!/usr/bin/env python3
# Hypothesis: 1d Camarilla R3/S3 breakout with 1w EMA50 trend filter and volume confirmation.
# Long when price breaks above R3 with close > EMA50 (bullish trend) and volume > 1.5x average.
# Short when price breaks below S3 with close < EMA50 (bearish trend) and volume > 1.5x average.
# Uses discrete sizing 0.25. Target: 30-100 total trades over 4 years (7-25/year) on 1d timeframe.
# Camarilla levels from prior day provide institutional support/resistance. EMA50 filter ensures trend alignment.
# Volume spike confirms institutional participation. Works in bull markets via upward breaks and bear markets via downward breaks.

name = "1d_Camarilla_R3_S3_Breakout_1wEMA50_VolumeSpike_v1"
timeframe = "1d"
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
    
    # Calculate Camarilla levels from previous day's OHLC
    # Need at least 2 days: 1 for previous, 1 for current
    if n < 2:
        return np.zeros(n)
    
    # Previous day's OHLC (shifted by 1)
    high_prev = np.roll(high, 1)
    low_prev = np.roll(low, 1)
    close_prev = np.roll(close, 1)
    # Set first value to NaN as there is no previous day
    high_prev[0] = np.nan
    low_prev[0] = np.nan
    close_prev[0] = np.nan
    
    # Camarilla R3 and S3 levels
    camarilla_range = high_prev - low_prev
    r3 = close_prev + 1.1 * camarilla_range / 2
    s3 = close_prev - 1.1 * camarilla_range / 2
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate EMA50 on 1w data
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w EMA50 to 1d timeframe (wait for 1w bar to close)
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):  # Start from 1 since we use previous day's data
        # Skip if any required data is NaN
        if (np.isnan(r3[i]) or np.isnan(s3[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R3 with close > EMA50 (bullish trend) and volume spike
            if (close[i] > r3[i] and 
                close[i] > ema_50_aligned[i] and 
                volume[i] > 1.5 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S3 with close < EMA50 (bearish trend) and volume spike
            elif (close[i] < s3[i] and 
                  close[i] < ema_50_aligned[i] and 
                  volume[i] > 1.5 * avg_volume[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below S3 (reversal signal)
            if close[i] < s3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above R3 (reversal signal)
            if close[i] > r3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals