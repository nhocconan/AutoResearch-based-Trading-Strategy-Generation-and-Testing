#!/usr/bin/env python3
"""
4h_Keltner_Breakout_Trend_Volume
Hypothesis: Price breaking above/below Keltner Channel (EMA20 +/- ATR*2) with trend filter (EMA50) and volume confirmation (volume > 1.5x 20-period average).
In bull markets, breaks above upper band signal continuation; in bear markets, breaks below lower band signal continuation.
Volume confirms institutional participation. Trend filter avoids counter-trend whipsaws.
Designed for 4h timeframe to limit trade frequency and reduce fee drag.
Target: 50-150 trades over 4 years (12-37/year).
"""

name = "4h_Keltner_Breakout_Trend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4H INDICATORS (calculated on primary timeframe) ===
    # EMA20 for Keltner base
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    # ATR for channel width
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    upper_keltner = ema20 + 2 * atr
    lower_keltner = ema20 - 2 * atr
    
    # EMA50 for trend filter
    ema50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume average (20-period)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === 1D TREND FILTER (HTF) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (max of all lookbacks)
    start_idx = max(50, 20)  # EMA50 needs 50, EMA20 needs 20
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(ema50[i]) or np.isnan(upper_keltner[i]) or 
            np.isnan(lower_keltner[i]) or np.isnan(vol_ma20[i]) or
            np.isnan(ema50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above upper Keltner, above EMA50 trend, volume spike
            if (close[i] > upper_keltner[i] and 
                close[i] > ema50[i] and 
                close[i] > ema50_1d_aligned[i] and 
                volume[i] > 1.5 * vol_ma20[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Keltner, below EMA50 trend, volume spike
            elif (close[i] < lower_keltner[i] and 
                  close[i] < ema50[i] and 
                  close[i] < ema50_1d_aligned[i] and 
                  volume[i] > 1.5 * vol_ma20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price closes below EMA20 (trend change) OR opposite Keltner break with volume
            if close[i] < ema20[i] or \
               (close[i] < lower_keltner[i] and volume[i] > 1.5 * vol_ma20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price closes above EMA20 OR opposite Keltner break with volume
            if close[i] > ema20[i] or \
               (close[i] > upper_keltner[i] and volume[i] > 1.5 * vol_ma20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals