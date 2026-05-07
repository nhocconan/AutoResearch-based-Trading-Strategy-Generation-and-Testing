#!/usr/bin/env python3
name = "4h_Camarilla_R3_S3_Breakout_1dTrend_Volume_Spike"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE for Camarilla levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Previous day's high, low, close for Camarilla calculation
    prev_high = df_1d['high'].values
    prev_low = df_1d['low'].values
    prev_close = df_1d['close'].values
    
    # Calculate Camarilla levels (R3, S3)
    # R3 = close + (high - low) * 1.1 / 4
    # S3 = close - (high - low) * 1.1 / 4
    camarilla_r3 = prev_close + (prev_high - prev_low) * 1.1 / 4
    camarilla_s3 = prev_close - (prev_high - prev_low) * 1.1 / 4
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r3_4h = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_4h = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Daily EMA34 for trend filter
    ema_34_1d = pd.Series(prev_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike detection: current volume > 1.5 * 20-period average volume
    vol_series = pd.Series(volume)
    vol_ma_20 = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Ensure EMA34 and volume MA are valid
    
    for i in range(start_idx, n):
        if (np.isnan(camarilla_r3_4h[i]) or np.isnan(camarilla_s3_4h[i]) or 
            np.isnan(ema_34_4h[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R3 + daily uptrend + volume spike
            if close[i] > camarilla_r3_4h[i] and close[i] > ema_34_4h[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 + daily downtrend + volume spike
            elif close[i] < camarilla_s3_4h[i] and close[i] < ema_34_4h[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price breaks below S3 or trend reversal
            if close[i] < camarilla_s3_4h[i] or close[i] < ema_34_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price breaks above R3 or trend reversal
            if close[i] > camarilla_r3_4h[i] or close[i] > ema_34_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals