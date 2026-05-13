#!/usr/bin/env python3
# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation.
# Long when price breaks above R3 with EMA34 rising and volume > 1.8x average.
# Short when price breaks below S3 with EMA34 falling and volume > 1.8x average.
# Uses discrete sizing 0.25. Target: 75-200 total trades over 4 years (19-50/year) on 4h timeframe.
# EMA34 on daily timeframe provides smooth trend direction with less whipsaw than ADX.
# Volume spike confirms institutional participation. Works in bull markets (long R3 breaks) and bear markets (short S3 breaks).

name = "4h_Camarilla_R3_S3_Breakout_1dEMA34_VolumeSpike_v1"
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
    
    # Calculate Camarilla levels from previous day (6*4h ≈ 24h)
    lookback = 6
    if n < lookback + 1:
        return np.zeros(n)
    
    high_prev = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().shift(1).values
    low_prev = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().shift(1).values
    close_prev = pd.Series(close).rolling(window=lookback, min_periods=lookback).mean().shift(1).values
    
    camarilla_range = high_prev - low_prev
    r3 = close_prev + 1.1 * camarilla_range / 2
    s3 = close_prev - 1.1 * camarilla_range / 2
    
    # Average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA34 on 1d close
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(lookback + 20, n):
        # Skip if any required data is NaN
        if (np.isnan(r3[i]) or np.isnan(s3[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R3 with EMA34 rising (trend up) and volume spike
            if (i > 0 and 
                close[i] > r3[i] and 
                ema_34_aligned[i] > ema_34_aligned[i-1] and 
                volume[i] > 1.8 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S3 with EMA34 falling (trend down) and volume spike
            elif (i > 0 and 
                  close[i] < s3[i] and 
                  ema_34_aligned[i] < ema_34_aligned[i-1] and 
                  volume[i] > 1.8 * avg_volume[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below S3 (reversal) OR EMA34 flattens/falls
            if (close[i] < s3[i]) or (i > 0 and ema_34_aligned[i] <= ema_34_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above R3 (reversal) OR EMA34 flattens/rises
            if (close[i] > r3[i]) or (i > 0 and ema_34_aligned[i] >= ema_34_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals