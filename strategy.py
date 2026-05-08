#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Camarilla_R3_S3_Breakout_1dTrend_Volume_Filtered"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for Camarilla levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate daily EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate ATR for volatility filter (14-period on daily)
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0  # first value has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate Camarilla levels from previous day
    R3 = np.zeros(len(close_1d))
    S3 = np.zeros(len(close_1d))
    
    for i in range(1, len(close_1d)):
        high_prev = high_1d[i-1]
        low_prev = low_1d[i-1]
        close_prev = close_1d[i-1]
        range_val = high_prev - low_prev
        
        if range_val <= 0:
            R3[i] = R3[i-1] if i > 1 else close_prev
            S3[i] = S3[i-1] if i > 1 else close_prev
        else:
            C = close_prev + (range_val * 1.1 / 6)
            R3[i] = C + (range_val * 1.1 / 2)
            S3[i] = C - (range_val * 1.1 / 2)
    
    # Forward fill first value
    if len(R3) > 0:
        R3[0] = R3[1] if len(R3) > 1 else close_1d[0]
        S3[0] = S3[1] if len(S3) > 1 else close_1d[0]
    
    # Align Camarilla levels to 4h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    # Volume spike: current volume > 2.0x 20-period average (stricter)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma20)
    
    # Volatility filter: ATR > 20-period average (avoid low volatility chop)
    atr_ma20 = pd.Series(atr_14_1d_aligned).rolling(window=20, min_periods=20).mean().values
    vol_filter = atr_14_1d_aligned > atr_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(R3_aligned[i]) or 
            np.isnan(S3_aligned[i]) or np.isnan(volume_spike[i]) or np.isnan(vol_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R3, daily uptrend, volume spike, sufficient volatility
            long_cond = (close[i] > R3_aligned[i] and 
                        ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1] and
                        volume_spike[i] and
                        vol_filter[i])
            
            # Short: price breaks below S3, daily downtrend, volume spike, sufficient volatility
            short_cond = (close[i] < S3_aligned[i] and 
                         ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1] and
                         volume_spike[i] and
                         vol_filter[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below S3
            if close[i] < S3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above R3
            if close[i] > R3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals