#!/usr/bin/env python3
# 4h_4h_1d_RSI_CCI_Coherence_v1
# Hypothesis: Uses RSI(14) and CCI(20) on 4h timeframe for momentum confirmation,
# combined with 1d CCI(20) trend filter to ensure trades align with higher timeframe momentum.
# Only enters when both timeframes show coherent momentum (both bullish or both bearish).
# Uses volume confirmation to filter false signals.
# Designed for low trade frequency by requiring triple confirmation (4h RSI, 4h CCI, 1d CCI).
# Works in both bull and bear markets by following the higher timeframe momentum.

name = "4h_4h_1d_RSI_CCI_Coherence_v1"
timeframe = "4h"
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
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 4h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 4h RSI(14) ---
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # --- 4h CCI(20) ---
    tp = (high + low + close) / 3.0
    sma_tp = pd.Series(tp).rolling(window=20, min_periods=20).mean().values
    mad = pd.Series(tp).rolling(window=20, min_periods=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True).values
    cci = (tp - sma_tp) / (0.015 * mad + 1e-10)
    
    # --- 1d CCI(20) for trend filter ---
    tp_1d = (df_1d['high'].values + df_1d['low'].values + df_1d['close'].values) / 3.0
    sma_tp_1d = pd.Series(tp_1d).rolling(window=20, min_periods=20).mean().values
    mad_1d = pd.Series(tp_1d).rolling(window=20, min_periods=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True).values
    cci_1d = (tp_1d - sma_tp_1d) / (0.015 * mad_1d + 1e-10)
    cci_1d_aligned = align_htf_to_ltf(prices, df_1d, cci_1d)
    
    # --- Volume Confirmation (20-period average) ---
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi[i]) or np.isnan(cci[i]) or np.isnan(cci_1d_aligned[i]) or
            np.isnan(vol_ratio[i])):
            # Maintain position if valid, otherwise flat
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation threshold
        volume_spike = vol_ratio[i] > 1.3
        
        if position == 0:
            # Long: 4h RSI > 50 (bullish momentum), 4h CCI > 0 (bullish), 1d CCI > 0 (bullish trend), with volume
            if (rsi[i] > 50 and 
                cci[i] > 0 and 
                cci_1d_aligned[i] > 0 and 
                volume_spike):
                signals[i] = 0.25
                position = 1
            # Short: 4h RSI < 50 (bearish momentum), 4h CCI < 0 (bearish), 1d CCI < 0 (bearish trend), with volume
            elif (rsi[i] < 50 and 
                  cci[i] < 0 and 
                  cci_1d_aligned[i] < 0 and 
                  volume_spike):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: loss of momentum coherence or volume
            if position == 1:
                # Exit long: loss of bullish coherence
                if (rsi[i] <= 50 or 
                    cci[i] <= 0 or 
                    cci_1d_aligned[i] <= 0):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: loss of bearish coherence
                if (rsi[i] >= 50 or 
                    cci[i] >= 0 or 
                    cci_1d_aligned[i] >= 0):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals