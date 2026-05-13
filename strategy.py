#!/usr/bin/env python3
# Hypothesis: 1d Williams %R reversal with 1w EMA50 trend filter and volume spike (>1.6x 20-bar avg). Uses 1d Camarilla R3/S3 levels for profit targets. Designed for BTC/ETH robustness: Williams %R captures overextended moves in both bull/bear regimes, 1w EMA50 ensures alignment with weekly trend, volume spike confirms participation. Targets 7-25 trades/year on 1d timeframe.

name = "1d_WilliamsR_Reversal_1wEMA50_VolumeSpike_CamarillaExits_v1"
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
    
    # Calculate 1w EMA50 for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Williams %R (14-period) on 1d data
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)  # avoid div by zero
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    
    # Calculate 1d Camarilla levels for dynamic exits
    camarilla_r3 = close + (high - low) * 1.1 / 4
    camarilla_s3 = close - (high - low) * 1.1 / 4
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # start after lookback
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(williams_r[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Williams %R oversold (< -80), price > 1w EMA50, volume spike (>1.6x avg)
            if (williams_r[i] < -80 and 
                close[i] > ema_50_1w_aligned[i] and 
                volume[i] > 1.6 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Williams %R overbought (> -20), price < 1w EMA50, volume spike (>1.6x avg)
            elif (williams_r[i] > -20 and 
                  close[i] < ema_50_1w_aligned[i] and 
                  volume[i] > 1.6 * avg_volume[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price reaches Camarilla R3 (profit target) OR Williams %R becomes overbought (> -20)
            if (close[i] >= camarilla_r3[i] or 
                williams_r[i] > -20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price reaches Camarilla S3 (profit target) OR Williams %R becomes oversold (< -80)
            if (close[i] <= camarilla_s3[i] or 
                williams_r[i] < -80):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals