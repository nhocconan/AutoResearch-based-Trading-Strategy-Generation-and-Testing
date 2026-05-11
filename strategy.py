#!/usr/bin/env python3
name = "4h_Keltner_Trend_Filter_12hEMA20_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter and ATR calculation
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12h EMA20 for trend filter
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=20, min_periods=20).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Calculate 12h ATR(10) for Keltner bands
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_12h = pd.Series(tr).ewm(span=10, min_periods=10).mean().values
    atr_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_12h)
    
    # Calculate Keltner Bands (20-period EMA ± 2*ATR) on 12h
    keltner_upper = ema_12h + (2 * atr_12h)
    keltner_lower = ema_12h - (2 * atr_12h)
    keltner_upper_aligned = align_htf_to_ltf(prices, df_12h, keltner_upper)
    keltner_lower_aligned = align_htf_to_ltf(prices, df_12h, keltner_lower)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_12h_aligned[i]) or np.isnan(keltner_upper_aligned[i]) or 
            np.isnan(keltner_lower_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above Keltner upper band AND above 12h EMA20 (uptrend) AND volume surge
            if close[i] > keltner_upper_aligned[i] and close[i] > ema_12h_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price below Keltner lower band AND below 12h EMA20 (downtrend) AND volume surge
            elif close[i] < keltner_lower_aligned[i] and close[i] < ema_12h_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price falls below Keltner lower band OR below 12h EMA20 (trend change)
            if close[i] < keltner_lower_aligned[i] or close[i] < ema_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price rises above Keltner upper band OR above 12h EMA20 (trend change)
            if close[i] > keltner_upper_aligned[i] or close[i] > ema_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals