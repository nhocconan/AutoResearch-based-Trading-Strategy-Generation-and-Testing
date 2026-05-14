#!/usr/bin/env python3
name = "4h_Chaikin_Money_Flow_Momentum_With_Volume_Filter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # CMF(20) calculation
    mfm = np.where((high - low) != 0, ((close - low) - (high - close)) / (high - low), 0)
    mfv = mfm * volume
    
    mfv_sum_20 = np.full(n, np.nan)
    vol_sum_20 = np.full(n, np.nan)
    for i in range(19, n):
        mfv_sum_20[i] = np.sum(mfv[i-19:i+1])
        vol_sum_20[i] = np.sum(volume[i-19:i+1])
    
    cmf = np.divide(mfv_sum_20, vol_sum_20, out=np.full_like(mfv_sum_20, np.nan), where=vol_sum_20!=0)
    
    # Daily trend filter using EMA(34)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume filter: current volume > 1.3 x 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        if (np.isnan(cmf[i]) or np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_condition = volume[i] > 1.3 * vol_ma_20[i]
        
        if position == 0:
            # LONG: CMF > 0.05 with bullish trend and volume
            if cmf[i] > 0.05 and close[i] > ema34_1d_aligned[i] and vol_condition:
                signals[i] = 0.25
                position = 1
            # SHORT: CMF < -0.05 with bearish trend and volume
            elif cmf[i] < -0.05 and close[i] < ema34_1d_aligned[i] and vol_condition:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: CMF < 0 or trend reversal
            if cmf[i] < 0 or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: CMF > 0 or trend reversal
            if cmf[i] > 0 or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals