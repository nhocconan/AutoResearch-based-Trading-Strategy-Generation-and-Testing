#!/usr/bin/env python3
"""
6h_1d_keltner_breakout_trend
Hypothesis: 6-hour Keltner breakout with daily trend filter. Uses daily EMA50 as trend filter and Keltner bands (ATR-based) for breakout signals. Works in both bull and bear by only taking long trades when above daily EMA50 and short trades when below. ATR filter prevents trading in low volatility. Target: 20-50 trades/year.
"""

name = "6h_1d_keltner_breakout_trend"
timeframe = "6h"
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
    
    # Get daily data for trend and Keltner calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Daily EMA50 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Daily ATR(10) for Keltner bands
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Daily EMA20 for Keltner center line
    ema20_1d = close_1d_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Keltner Bands: Upper = EMA20 + 2*ATR, Lower = EMA20 - 2*ATR
    keltner_upper = ema20_1d + 2.0 * atr_1d
    keltner_lower = ema20_1d - 2.0 * atr_1d
    
    # Align daily indicators to 6h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    keltner_upper_aligned = align_htf_to_ltf(prices, df_1d, keltner_upper)
    keltner_lower_aligned = align_htf_to_ltf(prices, df_1d, keltner_lower)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(keltner_upper_aligned[i]) or 
            np.isnan(keltner_lower_aligned[i]) or np.isnan(atr_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Long entry: close breaks above Keltner Upper with volume, above daily EMA50, and sufficient volatility
        if (close[i] > keltner_upper_aligned[i] and vol_confirm[i] and 
            close[i] > ema50_1d_aligned[i] and atr_1d_aligned[i] > 0 and position != 1):
            position = 1
            signals[i] = 0.25
        # Short entry: close breaks below Keltner Lower with volume, below daily EMA50, and sufficient volatility
        elif (close[i] < keltner_lower_aligned[i] and vol_confirm[i] and 
              close[i] < ema50_1d_aligned[i] and atr_1d_aligned[i] > 0 and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: reverse signal or close crosses back to opposite Keltner band
        elif position == 1 and close[i] < keltner_lower_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > keltner_upper_aligned[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals