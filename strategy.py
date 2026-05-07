#!/usr/bin/env python3
name = "12h_Camarilla_R3S3_Breakout_1dTrend_Volume"
timeframe = "12h"
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
    
    # Load daily data ONCE for Camarilla and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Daily high, low, close for Camarilla calculation
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    daily_volume = df_1d['volume'].values
    
    # Camarilla R3 and S3 levels from previous day
    # R3 = close + 1.1 * (high - low) / 2
    # S3 = close - 1.1 * (high - low) / 2
    camarilla_r3 = daily_close + 1.1 * (daily_high - daily_low) / 2
    camarilla_s3 = daily_close - 1.1 * (daily_high - daily_low) / 2
    
    # Align to 12h timeframe
    camarilla_r3_12h = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_12h = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Daily EMA34 for trend filter
    ema_34_1d = pd.Series(daily_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike detection on 12h
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Need EMA34 to be valid
    
    for i in range(start_idx, n):
        if (np.isnan(camarilla_r3_12h[i]) or np.isnan(camarilla_s3_12h[i]) or 
            np.isnan(ema_34_12h[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: volume > 1.5x average
        volume_spike = vol_ratio[i] > 1.5
        
        # Trend filter: price above/below EMA34
        uptrend = close[i] > ema_34_12h[i]
        downtrend = close[i] < ema_34_12h[i]
        
        if position == 0:
            # Long: price breaks above R3 + volume spike + uptrend
            if close[i] > camarilla_r3_12h[i] and volume_spike and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 + volume spike + downtrend
            elif close[i] < camarilla_s3_12h[i] and volume_spike and downtrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price breaks below S3 or trend reversal
            if close[i] < camarilla_s3_12h[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price breaks above R3 or trend reversal
            if close[i] > camarilla_r3_12h[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals