#!/usr/bin/env python3
# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume confirmation.
# Long when price breaks above Camarilla R3 level, 4h EMA50 rising, volume > 1.5x average.
# Short when price breaks below Camarilla S3 level, 4h EMA50 falling, volume > 1.5x average.
# Uses discrete sizing 0.20. Target: 60-150 total trades over 4 years (15-37/year) on 1h timeframe.
# Camarilla levels provide intraday support/resistance. EMA50 on 4h filters counter-trend trades.
# Volume confirmation ensures institutional breakouts. Works in bull markets via upward breaks
# and in bear markets via downward breaks. Session filter (08-20 UTC) reduces noise.

name = "1h_Camarilla_R3_S3_Breakout_4hEMA50_VolumeConfirm_v1"
timeframe = "1h"
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
    
    # Calculate Camarilla levels for current day (using previous day's range)
    # Camarilla: R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low)
    #          S3 = close - 1.1*(high-low), S4 = close - 1.5*(high-low)
    # We use daily high/low from 1d timeframe
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla R3 and S3 from previous 1d bar
    # R3 = close_1d + 1.1 * (high_1d - low_1d)
    # S3 = close_1d - 1.1 * (high_1d - low_1d)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d[0] = np.nan
    prev_low_1d[0] = np.nan
    prev_close_1d[0] = np.nan
    
    camarilla_r3 = prev_close_1d + 1.1 * (prev_high_1d - prev_low_1d)
    camarilla_s3 = prev_close_1d - 1.1 * (prev_high_1d - prev_low_1d)
    
    # Align Camarilla levels to 1h timeframe (wait for 1d bar to close)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Get 4h data for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA50 for trend filter
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 4h EMA50 to 1h timeframe (wait for 4h bar to close)
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour  # prices.index is DatetimeIndex
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after sufficient data for indicators
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema50_4h_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Camarilla R3, 4h EMA50 rising, volume > 1.5x average
            if (close[i] > camarilla_r3_aligned[i] and 
                i > 50 and ema50_4h_aligned[i] > ema50_4h_aligned[i-1] and  # Rising EMA50
                volume[i] > 1.5 * avg_volume[i]):
                signals[i] = 0.20
                position = 1
            # SHORT: Price breaks below Camarilla S3, 4h EMA50 falling, volume > 1.5x average
            elif (close[i] < camarilla_s3_aligned[i] and 
                  i > 50 and ema50_4h_aligned[i] < ema50_4h_aligned[i-1] and  # Falling EMA50
                  volume[i] > 1.5 * avg_volume[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below Camarilla S3 OR 4h EMA50 falling
            if (close[i] < camarilla_s3_aligned[i] or 
                i > 50 and ema50_4h_aligned[i] < ema50_4h_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price breaks above Camarilla R3 OR 4h EMA50 rising
            if (close[i] > camarilla_r3_aligned[i] or 
                i > 50 and ema50_4h_aligned[i] > ema50_4h_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals