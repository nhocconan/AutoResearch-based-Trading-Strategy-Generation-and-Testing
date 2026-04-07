#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4h Choppiness Index regime + 12h KAMA direction + volume confirmation
# Hypothesis: Trade only in trending markets (Chop < 38.2) in direction of 12h KAMA
# with volume confirmation. Avoids choppy markets and false breakouts.
# Works in bull via trend following, in bear via avoiding false signals during consolidation.
# Target: 20-50 trades/year to minimize fee drag.
name = "4h_chop_kama_12h_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for KAMA direction
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 12h KAMA (using close prices)
    close_12h = df_12h['close'].values
    # Calculate Efficiency Ratio for KAMA
    change = np.abs(np.diff(close_12h, prepend=close_12h[0]))
    volatility = np.abs(np.diff(close_12h))
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (0.6645 - 0.0645) + 0.0645) ** 2
    # Calculate KAMA
    kama = np.zeros_like(close_12h)
    kama[0] = close_12h[0]
    for i in range(1, len(close_12h)):
        kama[i] = kama[i-1] + sc[i] * (close_12h[i] - kama[i-1])
    kama_12h = kama
    
    # Calculate 12h KAMA direction (1 if price > KAMA, -1 otherwise)
    kama_dir = np.where(close_12h > kama_12h, 1, -1)
    kama_dir_aligned = align_htf_to_ltf(prices, df_12h, kama_dir)
    
    # Calculate daily 20-period volume moving average
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Calculate Choppiness Index (14-period) on 4h data
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], 
                         np.maximum(tr1, np.maximum(tr2, tr3))])
    # Sum of True Range over 14 periods
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    # Absolute price change over 14 periods
    price_change = np.abs(np.subtract(close[14:], close[:-14]))
    price_change = np.concatenate([np.full(14, np.nan), price_change])
    # Choppiness Index
    chop = np.where(atr_sum != 0, 100 * np.log10(price_change / atr_sum) / np.log10(14), 50)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(14, n):
        # Skip if required data not available
        if (np.isnan(kama_dir_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]) or 
            np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > daily average volume
        vol_confirm = volume[i] > vol_ma_1d_aligned[i]
        
        # Regime filter: only trade when market is trending (Chop < 38.2)
        trending_regime = chop[i] < 38.2
        
        if position == 1:  # Long position
            # Exit: trend ends OR volatility increases (Chop > 61.8) OR volume confirmation lost
            if kama_dir_aligned[i] != 1 or not trending_regime or not vol_confirm:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: trend ends OR volatility increases (Chop > 61.8) OR volume confirmation lost
            if kama_dir_aligned[i] != -1 or not trending_regime or not vol_confirm:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Enter long: KAMA up + trending regime + volume confirmation
            if kama_dir_aligned[i] == 1 and trending_regime and vol_confirm:
                position = 1
                signals[i] = 0.25
            # Enter short: KAMA down + trending regime + volume confirmation
            elif kama_dir_aligned[i] == -1 and trending_regime and vol_confirm:
                position = -1
                signals[i] = -0.25
    
    return signals