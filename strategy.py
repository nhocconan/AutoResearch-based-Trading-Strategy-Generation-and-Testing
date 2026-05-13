#!/usr/bin/env python3
# Hypothesis: 6h Williams %R reversal with 12h EMA34 trend filter and volume spike (>1.8x 20-bar avg). Uses Camarilla R3/S3 levels from 1d for dynamic stop/profit targets. Designed for BTC/ETH robustness in both bull/bear regimes: Williams %R captures overextended moves, EMA34 filter ensures trend alignment, volume spike confirms institutional interest, and Camarilla levels provide structured exits. Targets 12-37 trades/year on 6f timeframe.

name = "6h_WilliamsR_Reversal_12hEMA34_VolumeSpike_CamarillaExits_v1"
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
    
    # Calculate 12h EMA34 for trend filter (HTF)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    close_12h = df_12h['close'].values
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Calculate Williams %R (14-period) on 6h data
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)  # avoid div by zero
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    
    # Calculate 1d Camarilla levels for dynamic exits (using HTF close)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    camarilla_r3 = close_1d + (high_1d - low_1d) * 1.1 / 4
    camarilla_s3 = close_1d - (high_1d - low_1d) * 1.1 / 4
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # start after lookback
        # Skip if any required data is NaN
        if (np.isnan(ema_34_12h_aligned[i]) or 
            np.isnan(williams_r[i]) or 
            np.isnan(avg_volume[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Williams %R oversold (< -80), price > 12h EMA34, volume spike (>1.8x avg)
            if (williams_r[i] < -80 and 
                close[i] > ema_34_12h_aligned[i] and 
                volume[i] > 1.8 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Williams %R overbought (> -20), price < 12h EMA34, volume spike (>1.8x avg)
            elif (williams_r[i] > -20 and 
                  close[i] < ema_34_12h_aligned[i] and 
                  volume[i] > 1.8 * avg_volume[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price reaches Camarilla R3 (profit target) OR Williams %R becomes overbought (> -20)
            if (close[i] >= camarilla_r3_aligned[i] or 
                williams_r[i] > -20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price reaches Camarilla S3 (profit target) OR Williams %R becomes oversold (< -80)
            if (close[i] <= camarilla_s3_aligned[i] or 
                williams_r[i] < -80):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals