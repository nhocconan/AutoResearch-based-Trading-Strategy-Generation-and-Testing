#!/usr/bin/env python3
name = "4h_Keltner_Breakout_1dTrend_VolumeSqueeze"
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
    
    # Get 1d data for trend filter and volatility (ATR)
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d EMA20 for trend filter
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=20, min_periods=20).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate 1d ATR(14) for Keltner channels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = np.maximum(high_1d - low_1d, np.abs(high_1d - np.roll(close_1d, 1)))
    tr2 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, tr2)
    tr[0] = high_1d[0] - low_1d[0]  # first period
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate 4h EMA20 for Keltner center line
    ema_20 = pd.Series(close).ewm(span=20, min_periods=20).mean().values
    
    # Calculate Keltner upper and lower bands (2x ATR multiplier)
    keltner_upper = ema_20 + 2 * atr_1d_aligned
    keltner_lower = ema_20 - 2 * atr_1d_aligned
    
    # Volume squeeze: current volume < 0.5x 20-period average (low volatility breakout setup)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_squeeze = volume < (vol_ma * 0.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(keltner_upper[i]) or np.isnan(keltner_lower[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(volume_squeeze[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above upper Keltner band AND above 1d EMA20 (uptrend) AND volume squeeze (volatility contraction before expansion)
            if close[i] > keltner_upper[i] and close[i] > ema_1d_aligned[i] and volume_squeeze[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Keltner band AND below 1d EMA20 (downtrend) AND volume squeeze
            elif close[i] < keltner_lower[i] and close[i] < ema_1d_aligned[i] and volume_squeeze[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price falls below EMA20 OR below lower Keltner band (mean reversion or trend change)
            if close[i] < ema_20[i] or close[i] < keltner_lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price rises above EMA20 OR above upper Keltner band (mean reversion or trend change)
            if close[i] > ema_20[i] or close[i] > keltner_upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals