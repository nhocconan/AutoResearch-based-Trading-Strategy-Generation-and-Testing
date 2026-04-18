#!/usr/bin/env python3
"""
1h_4h1d_Volume_Momentum_Regime_Filter
Hypothesis: Uses 4h momentum (close > SMA50) for directional bias, 1h volume surge (volume > 2x 20-period average) for entry timing, and daily volatility regime (ATR < 20-period ATR mean) to avoid choppy markets. Designed to capture momentum bursts in both bull and bear markets while filtering false signals. Target: 20-40 trades/year to minimize fee drag.
"""

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
    
    # Get 4h and daily data for multi-timeframe analysis
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h SMA50 for trend bias
    close_4h = df_4h['close'].values
    sma50_4h = np.full(len(close_4h), np.nan)
    for i in range(50, len(close_4h)):
        sma50_4h[i] = np.mean(close_4h[i-50:i])
    sma50_4h_aligned = align_htf_to_ltf(prices, df_4h, sma50_4h)
    
    # Calculate daily ATR for volatility regime
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]), np.abs(low_1d[1:] - close_1d[:-1]))
    tr1 = np.concatenate([[np.nan], tr1])
    atr_1d = np.full(len(tr1), np.nan)
    for i in range(20, len(tr1)):
        atr_1d[i] = np.mean(tr1[i-20:i])
    atr_mean_1d = np.full(len(atr_1d), np.nan)
    for i in range(20, len(atr_1d)):
        atr_mean_1d[i] = np.mean(atr_1d[i-20:i])
    low_vol_regime = atr_1d < atr_mean_1d  # True when volatility is below average
    low_vol_regime_aligned = align_htf_to_ltf(prices, df_1d, low_vol_regime.astype(float))
    
    # 1h volume confirmation: current volume > 2x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_confirm = volume > (vol_ma * 2.0)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(sma50_4h_aligned[i]) or np.isnan(low_vol_regime_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if not session_filter[i]:
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: price above 4h SMA50, volume surge, low volatility regime
            if (close[i] > sma50_4h_aligned[i] and vol_confirm[i] and 
                low_vol_regime_aligned[i] > 0.5):
                signals[i] = 0.20
                position = 1
            # Short entry: price below 4h SMA50, volume surge, low volatility regime
            elif (close[i] < sma50_4h_aligned[i] and vol_confirm[i] and 
                  low_vol_regime_aligned[i] > 0.5):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price crosses below 4h SMA50 or volatility increases
            if (close[i] < sma50_4h_aligned[i] or low_vol_regime_aligned[i] <= 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: price crosses above 4h SMA50 or volatility increases
            if (close[i] > sma50_4h_aligned[i] or low_vol_regime_aligned[i] <= 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_4h1d_Volume_Momentum_Regime_Filter"
timeframe = "1h"
leverage = 1.0