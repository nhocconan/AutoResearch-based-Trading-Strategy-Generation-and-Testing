#!/usr/bin/env python3
"""
Hypothesis: 6-hour 1-week Relative Strength Index (RSI) divergence combined with 1-day volume surge.
Long when weekly RSI shows bullish divergence (price makes lower low, RSI makes higher low) and daily volume > 1.5x 20-period average.
Short when weekly RSI shows bearish divergence (price makes higher high, RSI makes lower high) and daily volume > 1.5x 20-period average.
Exit when divergence breaks down or volume normalizes.
Designed for low frequency (~20-40/year) to capture major reversals in both bull and bear markets with minimal whipsaws.
"""

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
    
    # Load 1-week data for RSI divergence - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Load 1-day data for volume filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1-week RSI (14-period)
    close_1w = df_1w['close'].values
    delta = np.diff(close_1w, prepend=close_1w[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean()
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Calculate 1-day volume ratio (current / 20-period average)
    volume_1d = df_1d['volume'].values
    vol_ma = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean()
    volume_ratio = volume_1d / vol_ma
    volume_ratio_values = volume_ratio.values
    
    # Align HTF indicators to 6h timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_1w, rsi_values)
    volume_ratio_aligned = align_htf_to_ltf(prices, df_1d, volume_ratio_values)
    
    # Calculate weekly price swings for divergence detection
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Find weekly swing highs and lows
    def find_swings(arr, window=3):
        # Simple peak/trough detection
        peaks = np.zeros_like(arr, dtype=bool)
        troughs = np.zeros_like(arr, dtype=bool)
        for i in range(window, len(arr) - window):
            if arr[i] == np.max(arr[i-window:i+window+1]):
                peaks[i] = True
            if arr[i] == np.min(arr[i-window:i+window+1]):
                troughs[i] = True
        return peaks, troughs
    
    high_peaks, _ = find_swings(high_1w)
    _, low_troughs = find_swings(low_1w)
    
    # Align swing points to 6h timeframe
    high_peaks_aligned = align_htf_to_ltf(prices, df_1w, high_peaks.astype(float))
    low_troughs_aligned = align_htf_to_ltf(prices, df_1w, low_troughs.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Track divergence state
    bullish_div = False
    bearish_div = False
    last_rsi_low = 100
    last_rsi_high = 0
    last_price_low = np.inf
    last_price_high = 0
    
    for i in range(30, n):
        # Skip if data not ready
        if (np.isnan(rsi_aligned[i]) or np.isnan(volume_ratio_aligned[i]) or 
            np.isnan(high_peaks_aligned[i]) or np.isnan(low_troughs_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bullish_div = bearish_div = False
            continue
        
        rsi_val = rsi_aligned[i]
        vol_ratio = volume_ratio_aligned[i]
        is_peak = high_peaks_aligned[i] > 0.5
        is_trough = low_troughs_aligned[i] > 0.5
        
        # Update divergence tracking
        if is_trough:
            if rsi_val < last_rsi_low and low[i] < last_price_low:
                bearish_div = True  # Price makes lower low, RSI makes higher low -> bearish divergence?
                # Actually: price lower low + RSI higher low = bullish divergence
                # Price higher high + RSI lower high = bearish divergence
                if rsi_val > last_rsi_low and low[i] < last_price_low:
                    bullish_div = True
                elif rsi_val < last_rsi_high and high[i] > last_price_high:
                    bearish_div = True
            last_rsi_low = rsi_val
            last_price_low = low[i]
        
        if is_peak:
            if rsi_val > last_rsi_high and high[i] > last_price_high:
                bullish_div = True  # Price makes higher high, RSI makes lower high -> bearish divergence
            elif rsi_val < last_rsi_low and low[i] < last_price_low:
                bearish_div = True
            last_rsi_high = rsi_val
            last_price_high = high[i]
        
        # Reset divergence if RSI moves back to neutral territory
        if 40 < rsi_val < 60:
            bullish_div = bearish_div = False
        
        if position == 0:
            # Long: Bullish divergence + volume surge
            if bullish_div and vol_ratio > 1.5:
                signals[i] = 0.25
                position = 1
                bullish_div = False  # Reset after entry
            # Short: Bearish divergence + volume surge
            elif bearish_div and vol_ratio > 1.5:
                signals[i] = -0.25
                position = -1
                bearish_div = False  # Reset after entry
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Bearish divergence appears or volume normalizes
                if bearish_div or vol_ratio < 1.2:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Bullish divergence appears or volume normalizes
                if bullish_div or vol_ratio < 1.2:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                bullish_div = bearish_div = False
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_WeeklyRSI_Divergence_VolumeSurge"
timeframe = "6h"
leverage = 1.0