#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Keltner_Breakout_Trend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma20)
    
    # 1d data for trend and Keltner calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1d EMA20 for trend filter
    ema_20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # 1d ATR(10) for Keltner channels
    tr1 = np.maximum(high_1d[1:], low_1d[:-1]) - np.minimum(high_1d[1:], low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.max([high_1d[0] - low_1d[0], np.abs(high_1d[0] - close_1d[0]), np.abs(low_1d[0] - close_1d[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_10_1d = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Keltner channels: EMA20 ± 2*ATR
    keltner_upper = ema_20_1d + 2.0 * atr_10_1d
    keltner_lower = ema_20_1d - 2.0 * atr_10_1d
    
    # Align 1d indicators to 4h timeframe
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    keltner_upper_aligned = align_htf_to_ltf(prices, df_1d, keltner_upper)
    keltner_lower_aligned = align_htf_to_ltf(prices, df_1d, keltner_lower)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Sufficient warmup for EMA20 and ATR
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_20_1d_aligned[i]) or np.isnan(keltner_upper_aligned[i]) or 
            np.isnan(keltner_lower_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price above Keltner upper, above 1d EMA20, volume spike
            long_cond = (close[i] > keltner_upper_aligned[i]) and (close[i] > ema_20_1d_aligned[i]) and volume_spike[i]
            # Short conditions: price below Keltner lower, below 1d EMA20, volume spike
            short_cond = (close[i] < keltner_lower_aligned[i]) and (close[i] < ema_20_1d_aligned[i]) and volume_spike[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below Keltner lower (reversion to mean)
            if close[i] < keltner_lower_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above Keltner upper (reversion to mean)
            if close[i] > keltner_upper_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals