#!/usr/bin/env python3
"""
Hypothesis: 1h EMA crossover with 4h trend filter and volume confirmation for 1h timeframe.
Long when 1h EMA12 crosses above EMA26 AND 4h close > 4h EMA50 AND volume > 1.5x 20-period average.
Short when 1h EMA12 crosses below EMA26 AND 4h close < 4h EMA50 AND volume > 1.5x 20-period average.
Exit when opposite EMA crossover occurs.
Uses discrete position sizing (0.20) to minimize fee drag and targets 15-35 trades/year per symbol.
Designed to work in both bull and bear markets by using 4h trend filter to avoid counter-trend trades.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1h EMA12 and EMA26
    ema12 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema26 = pd.Series(close).ewm(span=26, adjust=False, min_periods=26).mean().values
    
    # Calculate 4h EMA50 for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    ema50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(26, 50, 20)  # EMA26 needs 26, 4h EMA50 needs 50, vol MA needs 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema12[i]) or np.isnan(ema26[i]) or np.isnan(ema50_4h_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        ema50_4h_val = ema50_4h_aligned[i]
        
        if position == 0:
            # Long: EMA12 crosses above EMA26 AND 4h uptrend AND volume confirmation
            if ema12[i] > ema26[i] and ema12[i-1] <= ema26[i-1] and close[i] > ema50_4h_val and volume[i] > 1.5 * vol_ma_val:
                signals[i] = 0.20
                position = 1
            # Short: EMA12 crosses below EMA26 AND 4h downtrend AND volume confirmation
            elif ema12[i] < ema26[i] and ema12[i-1] >= ema26[i-1] and close[i] < ema50_4h_val and volume[i] > 1.5 * vol_ma_val:
                signals[i] = -0.20
                position = -1
        else:
            # Exit when opposite EMA crossover occurs
            exit_signal = False
            
            if position == 1 and ema12[i] < ema26[i] and ema12[i-1] >= ema26[i-1]:
                exit_signal = True
            elif position == -1 and ema12[i] > ema26[i] and ema12[i-1] <= ema26[i-1]:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1H_EMA12_EMA26_Crossover_4hEMA50_Trend_VolumeConfirmation"
timeframe = "1h"
leverage = 1.0