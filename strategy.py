#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_TwinPeak_Reversal"
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
    
    # Get 1d data once before loop
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Twin Peak (double top/bottom) pattern
    # Peak detection: local maxima/minima over 5 periods
    window = 5
    peak = np.zeros_like(high_1d, dtype=bool)
    trough = np.zeros_like(low_1d, dtype=bool)
    
    for i in range(window, len(high_1d) - window):
        # Check for peak (local maximum)
        if high_1d[i] == np.max(high_1d[i-window:i+window+1]):
            peak[i] = True
        # Check for trough (local minimum)
        if low_1d[i] == np.min(low_1d[i-window:i+window+1]):
            trough[i] = True
    
    # Twin peak signals: two peaks/troughs within 20 periods
    twin_peak_high = np.zeros_like(high_1d, dtype=bool)
    twin_peak_low = np.zeros_like(low_1d, dtype=bool)
    
    peak_indices = np.where(peak)[0]
    trough_indices = np.where(trough)[0]
    
    # Look for two peaks within 20 periods
    for i in range(len(peak_indices)-1):
        if peak_indices[i+1] - peak_indices[i] <= 20:
            twin_peak_high[peak_indices[i+1]] = True
    
    # Look for two troughs within 20 periods
    for i in range(len(trough_indices)-1):
        if trough_indices[i+1] - trough_indices[i] <= 20:
            twin_peak_low[trough_indices[i+1]] = True
    
    # Twin peak values: 1 for bearish signal (double top), -1 for bullish signal (double bottom)
    twin_peak_signal = np.zeros_like(close_1d)
    twin_peak_signal[twin_peak_high] = 1   # Bearish: potential top
    twin_peak_signal[twin_peak_low] = -1   # Bullish: potential bottom
    
    # Align twin peak signal to 12h timeframe
    twin_peak_aligned = align_htf_to_ltf(prices, df_1d, twin_peak_signal)
    
    # Volume filter: current volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        if np.isnan(twin_peak_aligned[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        tp_signal = twin_peak_aligned[i]
        
        # Volume filter
        volume_ok = vol > 1.8 * vol_ma
        
        if position == 0:
            # Long: Twin bottom (-1) with volume confirmation
            if tp_signal < -0.5 and volume_ok:
                signals[i] = 0.25
                position = 1
            # Short: Twin top (1) with volume confirmation
            elif tp_signal > 0.5 and volume_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Twin top signal appears or volume drops
            if tp_signal > 0.5 or not volume_ok:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Twin bottom signal appears or volume drops
            if tp_signal < -0.5 or not volume_ok:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals