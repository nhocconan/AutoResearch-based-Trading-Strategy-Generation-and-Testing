#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_WMA_Trend_VolumeBreakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # === 1d: Weighted Moving Average (WMA) for trend filter ===
    close_1d = df_1d['close'].values
    # Calculate WMA(20) for 1d: weighted average with linearly decreasing weights
    window_wma = 20
    weights = np.arange(1, window_wma + 1)
    wma_1d = np.zeros_like(close_1d)
    for i in range(window_wma - 1, len(close_1d)):
        wma_1d[i] = np.dot(close_1d[i - window_wma + 1:i + 1], weights) / weights.sum()
    
    # Align WMA to 4h
    wma_1d_aligned = align_htf_to_ltf(prices, df_1d, wma_1d)
    
    # === 4h: Indicators ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for stop loss
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(50, 20)  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Get values
        wma_val = wma_1d_aligned[i]
        current_vol_ma = vol_ma[i]
        current_volume = volume[i]
        current_close = close[i]
        current_atr = atr[i]
        
        # Skip if any value is NaN
        if (np.isnan(wma_val) or np.isnan(current_vol_ma) or 
            np.isnan(current_atr)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: current volume > 2.0x 20-period average
        vol_condition = current_volume > 2.0 * current_vol_ma
        
        if position == 0:
            # Long: close > 1d WMA (uptrend) + volume breakout
            if current_close > wma_val and vol_condition:
                signals[i] = 0.25
                position = 1
                entry_price = current_close
            # Short: close < 1d WMA (downtrend) + volume breakout
            elif current_close < wma_val and vol_condition:
                signals[i] = -0.25
                position = -1
                entry_price = current_close
        
        elif position == 1:
            # Long exit: close < 1d WMA OR stop loss
            if current_close < wma_val or current_close < entry_price - 2.0 * current_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: close > 1d WMA OR stop loss
            if current_close > wma_val or current_close > entry_price + 2.0 * current_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals