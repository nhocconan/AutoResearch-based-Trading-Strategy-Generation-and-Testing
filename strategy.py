#!/usr/bin/env python3
"""
12h_KAMA_Direction_1dTrend_Filter
Hypothesis: KAMA (Kaufman Adaptive Moving Average) adapts to market noise, providing a reliable trend signal.
In trending markets (price above/below 1-day EMA34), take KAMA direction with volume confirmation.
In ranging markets, avoid trades to reduce whipsaw. Uses 12-hour timeframe for execution with 1d trend filter.
Target: 15-25 trades per year (~60-100 over 4 years) with position size 0.25.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_KAMA_Direction_1dTrend_Filter"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # KAMA parameters
    er_length = 10
    fast_sc = 2 / (2 + 1)  # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    
    # Calculate Efficiency Ratio (ER)
    change = np.abs(np.diff(close, n=er_length))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)
    # Use pandas for rolling sum to handle NaNs properly
    volatility_series = pd.Series(np.abs(np.diff(close))).rolling(window=er_length, min_periods=1).sum().values
    # Prepend zeros for the first er_length elements
    volatility = np.concatenate([np.full(er_length, np.nan), volatility_series])
    er = np.where(volatility > 0, change / volatility, 0)
    
    # Calculate Smoothing Constant (SC)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.full(n, np.nan)
    kama[er_length] = close[er_length]  # Initialize
    for i in range(er_length + 1, n):
        if np.isnan(sc[i]) or np.isnan(kama[i-1]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # 1-day EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume ratio: current volume / 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.where(vol_ma > 0, volume / vol_ma, 1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, er_length + 10)  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        if (np.isnan(kama[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine market regime from 1-day EMA34
        uptrend_regime = close[i] > ema_34_1d_aligned[i]
        downtrend_regime = close[i] < ema_34_1d_aligned[i]
        ranging_regime = not (uptrend_regime or downtrend_regime)
        
        # Volume confirmation: volume > 1.5x average
        volume_confirm = vol_ratio[i] > 1.5
        
        if position == 0 and not ranging_regime:
            # Long: price above KAMA in uptrend + volume
            long_entry = (close[i] > kama[i]) and uptrend_regime and volume_confirm
            # Short: price below KAMA in downtrend + volume
            short_entry = (close[i] < kama[i]) and downtrend_regime and volume_confirm
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price crosses below KAMA or regime changes to ranging/downtrend
            if (close[i] < kama[i]) or (not uptrend_regime):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price crosses above KAMA or regime changes to ranging/uptrend
            if (close[i] > kama[i]) or (not downtrend_regime):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals