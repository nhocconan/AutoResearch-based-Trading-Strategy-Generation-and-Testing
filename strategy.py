#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # 1d data for ATR and volume
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 14-day ATR
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 10-day volume MA for volume filter
    vol_ma10 = pd.Series(volume_1d).rolling(window=10, min_periods=10).mean().values
    
    # 4h data for EMA trend
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=20, min_periods=20, adjust=False).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # 4h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(200, n):
        # Skip if data not ready
        if (np.isnan(atr_1d[i]) or np.isnan(vol_ma10[i]) or np.isnan(ema_4h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: current 4h volume > 1.5 * 10-day average volume (scaled to 4h)
        # Approximate: 1 day = 6 * 4h bars, so scale daily volume MA to 4h
        vol_threshold = vol_ma10[i] / 6.0  # Convert daily volume MA to approximate 4h
        vol_surge = volume[i] > 2.0 * vol_threshold
        
        if position == 0:
            # Long: Close > EMA20 + 0.5 * ATR + volume surge
            if close[i] > ema_4h_aligned[i] + 0.5 * atr_1d[i] and vol_surge:
                signals[i] = 0.25
                position = 1
            # Short: Close < EMA20 - 0.5 * ATR + volume surge
            elif close[i] < ema_4h_aligned[i] - 0.5 * atr_1d[i] and vol_surge:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price returns to EMA20
            if position == 1:
                if close[i] < ema_4h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > ema_4h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_EMA20_ATR_VolumeSurge_v1"
timeframe = "4h"
leverage = 1.0