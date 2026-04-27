#!/usr/bin/env python3
"""
1d_Chaikin_Money_Flow_FlowIndex_WeeklyTrend
Hypothesis: Use Chaikin Money Flow (CMF) for institutional flow detection on daily timeframe.
Combine with weekly trend filter (EMA21) to avoid counter-trend trades.
Add volume confirmation to reduce false signals.
Designed for 30-100 trades over 4 years (7-25/year) on BTC/ETH.
Works in bull via CMF>0.1 + price>weekly EMA21, bear via CMF<-0.1 + price<weekly EMA21.
"""

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
    
    # Calculate Chaikin Money Flow on 1d timeframe
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Money Flow Multiplier
    mfm = ((close_1d - low_1d) - (high_1d - close_1d)) / (high_1d - low_1d)
    mfm = np.where((high_1d - low_1d) == 0, 0, mfm)
    
    # Money Flow Volume
    mfv = mfm * volume_1d
    
    # CMF(20) = 20-period sum of MFV / 20-period sum of volume
    mfv_sum = pd.Series(mfv).rolling(window=20, min_periods=20).sum().values
    vol_sum = pd.Series(volume_1d).rolling(window=20, min_periods=20).sum().values
    cmf = mfv_sum / vol_sum
    cmf = np.where(vol_sum == 0, 0, cmf)
    
    # Weekly trend filter: EMA21
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_21 = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Align to daily timeframe
    cmf_aligned = align_htf_to_ltf(prices, df_1d, cmf)
    ema_21_aligned = align_htf_to_ltf(prices, df_1w, ema_21)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(cmf_aligned[i]) or np.isnan(ema_21_aligned[i])):
            signals[i] = 0.0
            continue
        
        cmf_val = cmf_aligned[i]
        ema_21_val = ema_21_aligned[i]
        
        # Volume confirmation: require volume > 20-day average
        vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_ok = volume[i] > vol_ma[i] if not np.isnan(vol_ma[i]) else False
        
        if position == 0:
            # Long: CMF > 0.1 AND price above weekly EMA21 AND volume confirmation
            if cmf_val > 0.1 and close[i] > ema_21_val and vol_ok:
                signals[i] = size
                position = 1
            # Short: CMF < -0.1 AND price below weekly EMA21 AND volume confirmation
            elif cmf_val < -0.1 and close[i] < ema_21_val and vol_ok:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: CMF < 0 OR price crosses below weekly EMA21
            if cmf_val < 0 or close[i] < ema_21_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: CMF > 0 OR price crosses above weekly EMA21
            if cmf_val > 0 or close[i] > ema_21_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_Chaikin_Money_Flow_FlowIndex_WeeklyTrend"
timeframe = "1d"
leverage = 1.0