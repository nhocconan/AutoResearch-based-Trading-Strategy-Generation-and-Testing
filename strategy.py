#!/usr/bin/env python3
# Hypothesis: 1d Williams %R reversal with 1w EMA34 trend filter and volume spike (>1.8x 20-bar avg). Uses 1w Williams %R for regime filter (extreme readings = fading momentum). Designed for BTC/ETH robustness in both bull/bear regimes: Williams %R captures overextended moves, EMA34 filter ensures trend alignment, volume spike confirms institutional interest, and 1w Williams %R avoids counter-trend entries. Targets 7-25 trades/year on 1d timeframe.

name = "1d_WilliamsR_Reversal_1wEMA34_VolumeSpike_RegimeFilter_v1"
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
    
    # Calculate 1w EMA34 for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate 1w Williams %R for regime filter (avoid counter-trend)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    highest_high_1w = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    lowest_low_1w = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    williams_r_1w = -100 * (highest_high_1w - close_1w) / (highest_high_1w - lowest_low_1w)
    williams_r_1w = np.where((highest_high_1w - lowest_low_1w) == 0, -50, williams_r_1w)
    williams_r_1w_aligned = align_htf_to_ltf(prices, df_1w, williams_r_1w)
    
    # Calculate Williams %R (14-period) on 1d data
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # start after lookback
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(williams_r_1w_aligned[i]) or 
            np.isnan(williams_r[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Williams %R oversold (< -80), price > 1w EMA34, volume spike (>1.8x avg), 1w Williams %R not extremely oversold (> -90) to avoid fading momentum
            if (williams_r[i] < -80 and 
                close[i] > ema_34_1w_aligned[i] and 
                volume[i] > 1.8 * avg_volume[i] and
                williams_r_1w_aligned[i] > -90):
                signals[i] = 0.25
                position = 1
            # SHORT: Williams %R overbought (> -20), price < 1w EMA34, volume spike (>1.8x avg), 1w Williams %R not extremely overbought (< -10) to avoid fading momentum
            elif (williams_r[i] > -20 and 
                  close[i] < ema_34_1w_aligned[i] and 
                  volume[i] > 1.8 * avg_volume[i] and
                  williams_r_1w_aligned[i] < -10):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price reaches 1w EMA34 (trend invalidation) OR Williams %R becomes overbought (> -20)
            if (close[i] <= ema_34_1w_aligned[i] or 
                williams_r[i] > -20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price reaches 1w EMA34 (trend invalidation) OR Williams %R becomes oversold (< -80)
            if (close[i] >= ema_34_1w_aligned[i] or 
                williams_r[i] < -80):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals