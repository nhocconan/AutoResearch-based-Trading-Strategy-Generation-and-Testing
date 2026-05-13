#!/usr/bin/env python3
# Hypothesis: 4h Camarilla R3/S3 breakout with 1d ADX25 trend filter and volume spike confirmation.
# Long when price breaks above R3 with 1d ADX > 25 and volume > 1.8x average.
# Short when price breaks below S3 with 1d ADX > 25 and volume > 1.8x average.
# Uses discrete sizing 0.25. Target: 75-200 total trades over 4 years (19-50/year) on 4h timeframe.
# Camarilla levels provide institutional support/resistance. 1d ADX ensures we trade only when trending.
# Volume spike confirms participation. Works in bull markets via upward breaks and in bear markets via downward breaks.

name = "4h_Camarilla_R3_S3_Breakout_1dADX25_VolumeSpike_v1"
timeframe = "4h"
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
    
    # Calculate Camarilla levels from previous day (approx using 6x 4h bars)
    lookback = 6  # 6 * 4h = 24h approx
    if n < lookback + 1:
        return np.zeros(n)
    
    # Calculate rolling max/min/close for previous "day"
    high_prev = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().shift(1).values
    low_prev = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().shift(1).values
    close_prev = pd.Series(close).rolling(window=lookback, min_periods=lookback).mean().shift(1).values
    
    # Camarilla R3 and S3 levels
    camarilla_range = high_prev - low_prev
    r3 = close_prev + 1.1 * camarilla_range / 2
    s3 = close_prev - 1.1 * camarilla_range / 2
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get 1d data for ADX25 trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX on 1d data (14-period)
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = max(high[i] - high[i-1], 0)
            minus_dm[i] = max(low[i-1] - low[i], 0)
            if plus_dm[i] == minus_dm[i]:
                plus_dm[i] = 0
                minus_dm[i] = 0
            elif plus_dm[i] < minus_dm[i]:
                plus_dm[i] = 0
            else:
                minus_dm[i] = 0
            
            tr[i] = max(high[i] - low[i], 
                       abs(high[i] - close[i-1]), 
                       abs(low[i] - close[i-1]))
        
        # Smooth TR, +DM, -DM
        atr = np.zeros_like(high)
        plus_di = np.zeros_like(high)
        minus_di = np.zeros_like(high)
        
        # First value
        atr[period-1] = np.nanmean(tr[1:period])
        plus_dm_sum = np.nansum(plus_dm[1:period])
        minus_dm_sum = np.nansum(minus_dm[1:period])
        
        if atr[period-1] != 0:
            plus_di[period-1] = 100 * (plus_dm_sum / atr[period-1])
            minus_di[period-1] = 100 * (minus_dm_sum / atr[period-1])
        
        # Wilder's smoothing
        for i in range(period, len(high)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            plus_dm_val = ((plus_di[i-1] * (period-1)) + (100 * plus_dm[i] / atr[i])) / period if atr[i] != 0 else 0
            minus_dm_val = ((minus_di[i-1] * (period-1)) + (100 * minus_dm[i] / atr[i])) / period if atr[i] != 0 else 0
            plus_di[i] = plus_dm_val
            minus_di[i] = minus_dm_val
        
        # Calculate DX and ADX
        dx = np.zeros_like(high)
        adx = np.zeros_like(high)
        
        for i in range(period, len(high)):
            if plus_di[i] + minus_di[i] != 0:
                dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
        
        # First ADX value
        adx[2*period-1] = np.nanmean(dx[period:2*period])
        
        # Wilder's smoothing for ADX
        for i in range(2*period, len(high)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    
    # Align 1d ADX to 4h timeframe (wait for 1d bar to close)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(lookback + 20, n):  # Start after sufficient data
        # Skip if any required data is NaN
        if (np.isnan(r3[i]) or np.isnan(s3[i]) or 
            np.isnan(adx_1d_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R3 with 1d ADX > 25 and volume spike
            if (close[i] > r3[i] and 
                adx_1d_aligned[i] > 25 and 
                volume[i] > 1.8 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S3 with 1d ADX > 25 and volume spike
            elif (close[i] < s3[i] and 
                  adx_1d_aligned[i] > 25 and 
                  volume[i] > 1.8 * avg_volume[i]):
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