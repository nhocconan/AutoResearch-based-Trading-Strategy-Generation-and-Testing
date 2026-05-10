#!/usr/bin/env python3
# 4h_Camarilla_R3S3_Breakout_1dTrend_Volume
# Hypothesis: Camarilla pivot levels (R3/S3) on 1d timeframe with trend filter from 1d EMA(34) and volume confirmation,
# executed on 4h timeframe for lower frequency and reduced fee drag. Captures institutional breakouts with trend alignment.
# Target: 20-40 trades/year (~80-160 total over 4 years) to stay within optimal trade frequency for 4h.
# Works in both bull and bear markets by following higher-timeframe trend.

name = "4h_Camarilla_R3S3_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d OHLC for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's range
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    range_1d = high_1d - low_1d
    
    # Camarilla R3 and S3 levels
    r3 = close_1d + (range_1d * 1.1 / 4)
    s3 = close_1d - (range_1d * 1.1 / 4)
    
    # Align Camarilla levels to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # 1d EMA trend filter (34-period)
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    # Session filter: 08:00-20:00 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient warmup for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R3, 1d EMA uptrend, volume confirmation, session active
            if (close[i] > r3_aligned[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume_filter[i] and 
                session_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3, 1d EMA downtrend, volume confirmation, session active
            elif (close[i] < s3_aligned[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume_filter[i] and 
                  session_filter[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price breaks below S3 OR 1d EMA turns down
            if (close[i] < s3_aligned[i] or 
                close[i] < ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price breaks above R3 OR 1d EMA turns up
            if (close[i] > r3_aligned[i] or 
                close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals