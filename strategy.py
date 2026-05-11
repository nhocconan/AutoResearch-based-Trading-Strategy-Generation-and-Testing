#!/usr/bin/env python3
name = "4h_ChaikinFlow_1dTrend_VolumeFilter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for EMA100 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 100:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema100_1d = pd.Series(close_1d).ewm(span=100, adjust=False, min_periods=100).mean().values
    ema100_1d_aligned = align_htf_to_ltf(prices, df_1d, ema100_1d)
    
    # Chaikin Money Flow (CMF) calculation on 4h
    # MFM = [(Close - Low) - (High - Close)] / (High - Low)
    # MFV = MFM * Volume
    # CMF = 20-period sum of MFV / 20-period sum of Volume
    hl_range = high - low
    hl_range = np.where(hl_range == 0, 1e-10, hl_range)  # avoid division by zero
    mfm = ((close - low) - (high - close)) / hl_range
    mfv = mfm * volume
    
    # 20-period sums for CMF
    mfv_sum = np.zeros(n)
    vol_sum = np.zeros(n)
    period = 20
    
    for i in range(period-1, n):
        mfv_sum[i] = np.sum(mfv[i-period+1:i+1])
        vol_sum[i] = np.sum(volume[i-period+1:i+1])
    
    cmf = np.zeros(n)
    cmf[period-1:] = mfv_sum[period-1:] / vol_sum[period-1:]
    
    # Volume filter: current volume > 1.5x 20-period average volume
    vol_ma = np.zeros(n)
    for i in range(period-1, n):
        vol_ma[i] = np.mean(volume[i-period+1:i+1])
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(100, period-1)  # Ensure EMA100 and CMF are ready
    
    for i in range(start_idx, n):
        if np.isnan(ema100_1d_aligned[i]) or np.isnan(cmf[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: CMF > 0.15, price above 1d EMA100, volume filter
            if (cmf[i] > 0.15 and 
                close[i] > ema100_1d_aligned[i] and 
                vol_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: CMF < -0.15, price below 1d EMA100, volume filter
            elif (cmf[i] < -0.15 and 
                  close[i] < ema100_1d_aligned[i] and 
                  vol_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: CMF < 0 or price below 1d EMA100
            if cmf[i] < 0 or close[i] < ema100_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: CMF > 0 or price above 1d EMA100
            if cmf[i] > 0 or close[i] > ema100_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals