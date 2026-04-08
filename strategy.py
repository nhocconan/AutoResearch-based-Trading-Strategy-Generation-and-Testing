#!/usr/bin/env python3
# 4h_1d_ema_crossover_volume_v4
# Hypothesis: Trade EMA crossovers on daily timeframe with volume confirmation on 4h.
# Uses EMA20/50 crossovers for trend detection, volume surge for confirmation, and ATR-based stops.
# Works in bull markets (trend following) and bear markets (counter-trend reversals at extremes).
# Target: 20-50 trades/year on 4h timeframe with strict entry conditions to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_ema_crossover_volume_v4"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily EMA for trend
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    ema20 = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align daily EMA to 4h timeframe
    ema20_aligned = align_htf_to_ltf(prices, df_1d, ema20)
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50)
    
    # ATR for volatility and stop
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation: 4h volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 100  # Ensure EMA50 and ATR are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema20_aligned[i]) or np.isnan(ema50_aligned[i]) or 
            np.isnan(atr[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Volume surge condition
        vol_surge = volume[i] > 1.5 * vol_ma_20[i] if vol_ma_20[i] > 0 else False
        
        if position == 1:  # Long position
            # Exit: EMA cross down OR stoploss hit
            if ema20_aligned[i] < ema50_aligned[i] or close[i] < ema20_aligned[i] - 2.5 * atr[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: EMA cross up OR stoploss hit
            if ema20_aligned[i] > ema50_aligned[i] or close[i] > ema20_aligned[i] + 2.5 * atr[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: EMA cross up with volume surge
            if ema20_aligned[i] > ema50_aligned[i] and vol_surge:
                position = 1
                signals[i] = 0.25
            # Short entry: EMA cross down with volume surge
            elif ema20_aligned[i] < ema50_aligned[i] and vol_surge:
                position = -1
                signals[i] = -0.25
    
    return signals