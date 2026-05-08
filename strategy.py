#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Chaikin_Momentum_WeeklyTrend"
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
    
    # Weekly trend filter: EMA34 on weekly close
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Daily data for Chaikin Money Flow (CMF) calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Money Flow Multiplier and Volume for CMF
    mfm = ((close_1d - low_1d) - (high_1d - close_1d)) / (high_1d - low_1d)
    mfm = np.where((high_1d - low_1d) == 0, 0, mfm)  # Avoid division by zero
    mfv = mfm * volume_1d
    
    # Calculate 20-period CMF
    mfv_sum = pd.Series(mfv).rolling(window=20, min_periods=20).sum().values
    volume_sum = pd.Series(volume_1d).rolling(window=20, min_periods=20).sum().values
    cmf_20 = np.divide(mfv_sum, volume_sum, out=np.zeros_like(mfv_sum), where=volume_sum!=0)
    
    # Align CMF to 6h timeframe
    cmf_20_aligned = align_htf_to_ltf(prices, df_1d, cmf_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(cmf_20_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: weekly uptrend + CMF > 0.1 (strong buying pressure)
            long_cond = (ema_34_1w_aligned[i] > ema_34_1w_aligned[i-1] and
                         cmf_20_aligned[i] > 0.1)
            
            # Short: weekly downtrend + CMF < -0.1 (strong selling pressure)
            short_cond = (ema_34_1w_aligned[i] < ema_34_1w_aligned[i-1] and
                          cmf_20_aligned[i] < -0.1)
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: weekly trend turns down OR CMF turns negative
            if (ema_34_1w_aligned[i] < ema_34_1w_aligned[i-1] or
                cmf_20_aligned[i] < 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: weekly trend turns up OR CMF turns positive
            if (ema_34_1w_aligned[i] > ema_34_1w_aligned[i-1] or
                cmf_20_aligned[i] > 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals