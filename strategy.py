#!/usr/bin/env python3
name = "6h_Keltner_Breakout_1dTrend_VolumeSpike"
timeframe = "6h"
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
    
    # Get 1d data for trend filter and volatility
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d ATR(14) for Keltner channels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d EMA(20) for Keltner middle line
    ema_20_1d = pd.Series(close_1d).ewm(span=20, min_periods=20).mean().values
    
    # Upper and lower Keltner bands (1d)
    keltner_upper = ema_20_1d + (2.0 * atr_1d)
    keltner_lower = ema_20_1d - (2.0 * atr_1d)
    
    # Align Keltner bands to 6h timeframe
    keltner_upper_6h = align_htf_to_ltf(prices, df_1d, keltner_upper)
    keltner_lower_6h = align_htf_to_ltf(prices, df_1d, keltner_lower)
    
    # Volume filter: current volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(keltner_upper_6h[i]) or np.isnan(keltner_lower_6h[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above upper Keltner band AND volume surge
            if close[i] > keltner_upper_6h[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Keltner band AND volume surge
            elif close[i] < keltner_lower_6h[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price falls below middle line (EMA20)
            if close[i] < keltner_upper_6h[i] - (atr_1d[i] * 2.0):  # using upper band as proxy for simplicity
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price rises above middle line (EMA20)
            if close[i] > keltner_lower_6h[i] + (atr_1d[i] * 2.0):  # using lower band as proxy for simplicity
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals