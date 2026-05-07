#!/usr/bin/env python3
name = "4h_Camarilla_R1S1_Breakout_1dEMA34_VolumeSpike"
timeframe = "4h"
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
    
    # 1d EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 1d Camarilla levels from previous day (high, low, close)
    # Camarilla R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # Since we don't have previous day's data in df_1d directly,
    # we'll use the current day's H,L,C to calculate levels for next day breakout
    # But to avoid look-ahead, we use previous day's data
    # We'll shift the 1d data by 1 to get previous day's OHLC
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Calculate Camarilla levels for previous day
    camarilla_R1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    camarilla_S1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    
    # Align Camarilla levels to 4h timeframe
    camarilla_R1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R1)
    camarilla_S1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S1)
    
    # Volume filter: volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(camarilla_R1_aligned[i]) or 
            np.isnan(camarilla_S1_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        volume_filter = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: Close breaks above Camarilla R1 in uptrend with volume
            if close[i] > camarilla_R1_aligned[i] and close[i] > ema_34_1d_aligned[i] and volume_filter:
                signals[i] = 0.25
                position = 1
            # Short: Close breaks below Camarilla S1 in downtrend with volume
            elif close[i] < camarilla_S1_aligned[i] and close[i] < ema_34_1d_aligned[i] and volume_filter:
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: Price reverts to opposite Camarilla level or trend fails
            if position == 1 and (close[i] < camarilla_S1_aligned[i] or close[i] < ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            elif position == -1 and (close[i] > camarilla_R1_aligned[i] or close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals