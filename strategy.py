#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ChaikinMoneyFlow_Momentum"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Chaikin Money Flow (volume flow)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Money Flow Multiplier for 1d
    mfm = ((close_1d - low_1d) - (high_1d - close_1d)) / (high_1d - low_1d)
    mfm = np.where((high_1d - low_1d) == 0, 0, mfm)  # avoid division by zero
    mfv = mfm * volume_1d
    
    # Chaikin Money Flow (20-period)
    mfv_sum = pd.Series(mfv).rolling(window=20, min_periods=20).sum().values
    volume_sum = pd.Series(volume_1d).rolling(window=20, min_periods=20).sum().values
    cmf = mfv_sum / volume_sum
    cmf = np.where(volume_sum == 0, 0, cmf)
    
    # Align CMF to 6h timeframe
    cmf_aligned = align_htf_to_ltf(prices, df_1d, cmf)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Weekly EMA34 for trend filter
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Volume spike filter on 6m: current volume > 1.8 * 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)  # Need enough data for CMF and volume MA
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(cmf_aligned[i]) or 
            np.isnan(ema34_1w_aligned[i]) or
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        cmf_val = cmf_aligned[i]
        ema34_1w = ema34_1w_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: CMF > 0.05 + weekly uptrend + volume spike
            if cmf_val > 0.05 and close[i] > ema34_1w and vol_spike:
                signals[i] = 0.25
                position = 1
            # Enter short: CMF < -0.05 + weekly downtrend + volume spike
            elif cmf_val < -0.05 and close[i] < ema34_1w and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: CMF falls below 0 or weekly trend turns down
            if cmf_val < 0 or close[i] < ema34_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: CMF rises above 0 or weekly trend turns up
            if cmf_val > 0 or close[i] > ema34_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals