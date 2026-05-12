#!/usr/bin/env python3
name = "4h_Keltner_Channel_Breakout_1dTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d Data for trend filter ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # === 1d EMA34 for trend ===
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # === Keltner Channel (4h) ===
    # ATR
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr_period = 20
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # EMA for middle line
    ema_period = 20
    ema = pd.Series(close).ewm(span=ema_period, adjust=False, min_periods=ema_period).mean().values
    
    multiplier = 2.0
    upper_band = ema + (multiplier * atr)
    lower_band = ema - (multiplier * atr)
    
    # === Volume spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(200, 34, 20)  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(ema[i]) or
            np.isnan(upper_band[i]) or
            np.isnan(lower_band[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above upper Keltner band + volume spike + 1d trend up
            if (close[i] > upper_band[i] and 
                volume_spike[i] and
                close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Keltner band + volume spike + 1d trend down
            elif (close[i] < lower_band[i] and 
                  volume_spike[i] and
                  close[i] < ema34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below EMA (middle line) or trend breaks
            if close[i] < ema[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above EMA (middle line) or trend breaks
            if close[i] > ema[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals