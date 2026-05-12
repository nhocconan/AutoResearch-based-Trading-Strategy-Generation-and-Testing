#!/usr/bin/env python3
name = "12h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike"
timeframe = "12h"
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
    
    # ===== 12h Close (LTF) =====
    # ===== Camarilla Pivot Levels from 1d (HTF) =====
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    camarilla_S3 = np.zeros(len(close_1d))
    camarilla_R3 = np.zeros(len(close_1d))
    camarilla_S4 = np.zeros(len(close_1d))
    camarilla_R4 = np.zeros(len(close_1d))
    
    for i in range(len(close_1d)):
        if i < 1:
            continue
        range_ = high_1d[i-1] - low_1d[i-1]
        camarilla_S3[i] = close_1d[i-1] - 1.1 * range_ / 6
        camarilla_R3[i] = close_1d[i-1] + 1.1 * range_ / 6
        camarilla_S4[i] = close_1d[i-1] - 1.1 * range_ / 4
        camarilla_R4[i] = close_1d[i-1] + 1.1 * range_ / 4
    
    # Align to 12h timeframe
    camarilla_S3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S3)
    camarilla_R3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R3)
    camarilla_S4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S4)
    camarilla_R4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R4)
    
    # ===== 1d Trend Filter (EMA34) =====
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # ===== 1d Volume Spike Filter =====
    vol_1d = df_1d['volume'].values
    vol_avg_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = vol_1d > (2.0 * vol_avg_1d)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d.astype(float))
    
    # ===== Session Filter: 08-20 UTC =====
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_R3_aligned[i]) or 
            np.isnan(camarilla_S3_aligned[i]) or
            np.isnan(ema34_1d_aligned[i]) or
            np.isnan(vol_spike_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Close breaks above R3 with volume spike + above daily EMA34
            if (close[i] > camarilla_R3_aligned[i] and
                vol_spike_1d_aligned[i] > 0.5 and
                close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Close breaks below S3 with volume spike + below daily EMA34
            elif (close[i] < camarilla_S3_aligned[i] and
                  vol_spike_1d_aligned[i] > 0.5 and
                  close[i] < ema34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Close below R3 or below S4 (strong reversal)
            if close[i] < camarilla_R3_aligned[i] or close[i] < camarilla_S4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Close above S3 or above R4 (strong reversal)
            if close[i] > camarilla_S3_aligned[i] or close[i] > camarilla_R4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals