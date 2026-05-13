#!/usr/bin/env python3
# Hypothesis: 1d Williams %R reversal with 1w EMA50 trend filter, volume spike (>1.8x 20-bar avg), and chop regime filter (CHOP(14) < 38.2 = trending). Uses discrete position sizing (0.25) to minimize fee drag. Williams %R identifies overbought/oversold conditions; 1w EMA50 ensures alignment with weekly trend; volume confirms breakout strength; chop filter avoids whipsaws in sideways markets. Designed for BTC/ETH robustness via confluence of momentum, weekly trend, volume, and regime filters.

name = "1d_WilliamsR_Reversal_1wEMA50_VolumeChopRegime_v1"
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
    
    # Calculate 1d Williams %R (14-period) for momentum reversal
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)  # avoid div by zero
    
    # Calculate 1d Choppiness Index (CHOP) on 14-period for regime filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    true_range = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(true_range).rolling(window=14, min_periods=14).mean().values
    atr_sum = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    true_range_sum = pd.Series(true_range).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(atr_sum / true_range_sum) / np.log10(14)
    chop = np.where(true_range_sum == 0, 50, chop)  # avoid div by zero
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # start after lookback
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(williams_r[i]) or 
            np.isnan(chop[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Williams %R oversold (< -80), price > 1w EMA50, volume spike (>1.8x avg), trending regime (CHOP < 38.2)
            if (williams_r[i] < -80 and 
                close[i] > ema_50_1w_aligned[i] and 
                volume[i] > 1.8 * avg_volume[i] and 
                chop[i] < 38.2):
                signals[i] = 0.25
                position = 1
            # SHORT: Williams %R overbought (> -20), price < 1w EMA50, volume spike (>1.8x avg), trending regime (CHOP < 38.2)
            elif (williams_r[i] > -20 and 
                  close[i] < ema_50_1w_aligned[i] and 
                  volume[i] > 1.8 * avg_volume[i] and 
                  chop[i] < 38.2):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close position if Williams %R becomes overbought (> -20) OR chop becomes too high (choppy market)
            if (williams_r[i] > -20 or 
                chop[i] > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close position if Williams %R becomes oversold (< -80) OR chop becomes too high (choppy market)
            if (williams_r[i] < -80 or 
                chop[i] > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals