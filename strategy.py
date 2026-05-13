#!/usr/bin/env python3
# Hypothesis: 12h Williams %R reversal with 1d EMA50 trend filter and volume confirmation (>1.5x 20-bar avg). Uses Camarilla R3/S3 levels from 1d for dynamic exits. Designed for BTC/ETH robustness: Williams %R captures overextended moves in both bull/bear regimes, EMA50 filter ensures trend alignment, volume spike confirms participation, and Camarilla levels provide structured profit targets. Targets 12-37 trades/year on 12h timeframe.

name = "12h_WilliamsR_Reversal_1dEMA50_VolumeConfirm_CamarillaExits_v1"
timeframe = "12h"
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
    
    # Calculate 1d EMA50 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Williams %R (14-period) on 12h data
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)  # avoid div by zero
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    
    # Calculate 1d Camarilla levels for dynamic exits
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    camarilla_r3 = close_1d + (high_1d - low_1d) * 1.1 / 4
    camarilla_s3 = close_1d - (high_1d - low_1d) * 1.1 / 4
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # start after lookback
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(williams_r[i]) or 
            np.isnan(avg_volume[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Williams %R oversold (< -80), price > 1d EMA50, volume spike (>1.5x avg)
            if (williams_r[i] < -80 and 
                close[i] > ema_50_1d_aligned[i] and 
                volume[i] > 1.5 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Williams %R overbought (> -20), price < 1d EMA50, volume spike (>1.5x avg)
            elif (williams_r[i] > -20 and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume[i] > 1.5 * avg_volume[i]):
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