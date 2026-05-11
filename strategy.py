#!/usr/bin/env python3
"""
6h_1d_Keltner_MeanReversion_With_Volume_Filter
Hypothesis: Mean reversion at Keltner Channel extremes (2.0 * ATR) on 6b timeframe,
filtered by 1d trend (EMA50) and volume spikes. Works in both bull and bear markets
by fading extremes only when aligned with higher timeframe trend, reducing whipsaw.
Targets ~20-30 trades/year (80-120 over 4 years) to minimize fee impact.
"""

name = "6h_1d_Keltner_MeanReversion_With_Volume_Filter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 6h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 1d Trend Filter: EMA50 ---
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # --- Keltner Channel (20, 2.0) on 6h ---
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    
    # ATR(20)
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    # EMA(20) of close
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Keltner Bands
    upper = ema20 + 2.0 * atr
    lower = ema20 - 2.0 * atr
    
    # --- Volume Filter: 20-period average ---
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50  # for EMA20, ATR, EMA50_1d, vol_ma
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema20[i]) or np.isnan(atr[i]) or 
            np.isnan(upper[i]) or np.isnan(lower[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine 1d trend
        trend_up = close[i] > ema50_1d_aligned[i]
        trend_down = close[i] < ema50_1d_aligned[i]
        
        # Volume confirmation
        vol_ok = volume[i] > vol_ma[i]
        
        if position == 0:
            # Look for mean reversion entries: price at Keltner extreme
            # Long: price at or below lower band + 1d uptrend + volume spike
            if close[i] <= lower[i] and trend_up and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short: price at or above upper band + 1d downtrend + volume spike
            elif close[i] >= upper[i] and trend_down and vol_ok:
                signals[i] = -0.25
                position = -1
        else:
            # Exit when price returns to EMA20 (mean)
            if position == 1:
                if close[i] >= ema20[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                if close[i] <= ema20[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals