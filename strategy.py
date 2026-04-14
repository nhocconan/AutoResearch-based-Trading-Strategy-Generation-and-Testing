#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ATR(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate 1d EMA200
    close_1d_series = pd.Series(close_1d)
    ema200_1d = close_1d_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Calculate 6h ATR(14) for volatility filter
    tr1_6h = high - low
    tr2_6h = np.abs(high - np.roll(close, 1))
    tr3_6h = np.abs(low - np.roll(close, 1))
    tr_6h = np.maximum(tr1_6h, np.maximum(tr2_6h, tr3_6h))
    tr_6h[0] = tr1_6h[0]
    atr_6h = pd.Series(tr_6h).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25
    
    for i in range(100, n):
        # Skip if any critical data is NaN
        if np.isnan(atr_1d_aligned[i]) or np.isnan(ema200_1d_aligned[i]):
            continue
        
        # Volatility filter: require 6h ATR > 0.5 * 1d ATR
        if atr_6h[i] < 0.5 * atr_1d_aligned[i]:
            continue
        
        # Get previous day's ATR for breakout threshold
        if i >= 1:
            prev_atr_1d = atr_1d_aligned[i-1] if not np.isnan(atr_1d_aligned[i-1]) else atr_1d_aligned[i]
            
            if position == 0:
                # Long: Close breaks above previous day's close + 0.5 * ATR
                if close[i] > close_1d_aligned[i-1] + 0.5 * prev_atr_1d and close[i] > ema200_1d_aligned[i]:
                    position = 1
                    signals[i] = position_size
                # Short: Close breaks below previous day's close - 0.5 * ATR
                elif close[i] < close_1d_aligned[i-1] - 0.5 * prev_atr_1d and close[i] < ema200_1d_aligned[i]:
                    position = -1
                    signals[i] = -position_size
            elif position == 1:
                # Exit: Close falls below EMA200 or ATR drops significantly
                if close[i] < ema200_1d_aligned[i] or atr_6h[i] < 0.3 * atr_1d_aligned[i]:
                    position = 0
                    signals[i] = 0.0
            elif position == -1:
                # Exit: Close rises above EMA200 or ATR drops significantly
                if close[i] > ema200_1d_aligned[i] or atr_6h[i] < 0.3 * atr_1d_aligned[i]:
                    position = 0
                    signals[i] = 0.0
    
    return signals

name = "6h_1d_ATR_Breakout_EMA200_Filter_v1"
timeframe = "6h"
leverage = 1.0