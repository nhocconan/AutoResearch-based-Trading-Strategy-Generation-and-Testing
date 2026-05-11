#!/usr/bin/env python3
name = "12h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike"
timeframe = "12h"
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
    
    # Get 1d data for trend filter and Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=34, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate 1d ATR(14) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = np.maximum(high_1d - low_1d, np.abs(high_1d - np.roll(close_1d, 1)))
    tr2 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, tr2)
    tr[0] = high_1d[0] - low_1d[0]  # first period
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate Camarilla pivot levels from previous 1d OHLC
    # Camarilla: R3 = close + (high - low) * 1.1/2, S3 = close - (high - low) * 1.1/2
    # R4 = close + (high - low) * 1.1, S4 = close - (high - low) * 1.1
    # We use R3/S3 as entry levels, R4/S4 as stop levels
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close[0] = close_1d[0]
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    
    camarilla_width = (prev_high - prev_low) * 1.1 / 2  # distance to R3/S3
    camarilla_r3 = prev_close + camarilla_width
    camarilla_s3 = prev_close - camarilla_width
    camarilla_r4 = prev_close + (prev_high - prev_low) * 1.1  # stop level
    camarilla_s4 = prev_close - (prev_high - prev_low) * 1.1  # stop level
    
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Volume spike: current volume > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(volume_spike[i]) or
            np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above Camarilla R3 AND above 1d EMA34 (uptrend) AND volume spike
            if close[i] > camarilla_r3_aligned[i] and close[i] > ema_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla S3 AND below 1d EMA34 (downtrend) AND volume spike
            elif close[i] < camarilla_s3_aligned[i] and close[i] < ema_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price falls below Camarilla S3 (reversal) OR below Camarilla S4 (stop) OR below 1d EMA34 (trend change)
            if close[i] < camarilla_s3_aligned[i] or close[i] < camarilla_s4_aligned[i] or close[i] < ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price rises above Camarilla R3 (reversal) OR above Camarilla R4 (stop) OR above 1d EMA34 (trend change)
            if close[i] > camarilla_r3_aligned[i] or close[i] > camarilla_r4_aligned[i] or close[i] > ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals