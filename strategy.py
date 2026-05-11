#!/usr/bin/env python3
"""
12h_Keltner_Channel_Breakout_1dTrend_Volume
Hypothesis: Price breaks above/below Keltner Channel (ATR-based bands) on 12h with 1d trend filter (price > 1d EMA50) and volume confirmation. Exits on opposite band touch. Designed for low trade frequency (~20-40/year) to work in both bull and bear markets by following 1d trend and avoiding whipsaws.
"""

name = "12h_Keltner_Channel_Breakout_1dTrend_Volume"
timeframe = "12h"
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
    
    # === 1D Data for Trend Filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # === 12h Keltner Channel (20, ATRx2) ===
    # EMA20 of close
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    # ATR(20)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = 0
    atr20 = pd.Series(tr).ewm(span=20, adjust=False, min_periods=20).mean().values
    # Keltner Bands
    kelner_upper = ema20 + 2 * atr20
    kelner_lower = ema20 - 2 * atr20
    
    # === Volume Filter: 1.5x 20-period EMA on 12h ===
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > vol_ema20 * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (covers 1d EMA50 and 12h EMA20/ATR20)
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(kelner_upper[i]) or np.isnan(kelner_lower[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above Keltner Upper with uptrend and volume spike
            if (close[i] > kelner_upper[i] and 
                close[i] > ema50_1d_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Keltner Lower with downtrend and volume spike
            elif (close[i] < kelner_lower[i] and 
                  close[i] < ema50_1d_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price touches or crosses below Keltner Lower (mean reversion)
            if close[i] < kelner_lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price touches or crosses above Keltner Upper (mean reversion)
            if close[i] > kelner_upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals