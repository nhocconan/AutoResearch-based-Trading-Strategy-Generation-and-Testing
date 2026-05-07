#!/usr/bin/env python3
name = "12h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike_v2"
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
    
    # Get daily data for Camarilla levels, trend, and volatility
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1d EMA34 trend filter (requires 34 periods)
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 1d ATR for volatility
    tr1 = np.maximum(df_1d['high'].values - df_1d['low'].values,
                     np.maximum(np.abs(df_1d['high'].values - np.roll(df_1d['close'].values, 1)),
                                np.abs(df_1d['low'].values - np.roll(df_1d['close'].values, 1))))
    tr1[0] = df_1d['high'].values[0] - df_1d['low'].values[0]
    atr_1d = pd.Series(tr1).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate Camarilla levels from previous day
    close_prev = np.roll(df_1d['close'].values, 1)
    high_prev = np.roll(df_1d['high'].values, 1)
    low_prev = np.roll(df_1d['low'].values, 1)
    close_prev[0] = df_1d['close'].values[0]
    high_prev[0] = df_1d['high'].values[0]
    low_prev[0] = df_1d['low'].values[0]
    
    R3 = close_prev + (high_prev - low_prev) * 1.1 / 4
    S3 = close_prev - (high_prev - low_prev) * 1.1 / 4
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    # Volume filter: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34)  # Need 20 for volume MA, 34 for EMA
    
    for i in range(start_idx, n):
        if np.isnan(ema_34_1d_aligned[i]) or np.isnan(atr_1d_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R3 with volume and above daily EMA34
            if close[i] > R3_aligned[i] and volume_ok[i] and close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 with volume and below daily EMA34
            elif close[i] < S3_aligned[i] and volume_ok[i] and close[i] < ema_34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: price returns to S3/R3 range or breaks in opposite direction
            if position == 1:
                if close[i] < S3_aligned[i] or close[i] < ema_34_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > R3_aligned[i] or close[i] > ema_34_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals