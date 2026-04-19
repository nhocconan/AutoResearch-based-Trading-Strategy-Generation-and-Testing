#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_Williams_Fractal_Trend_Volume_V1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams Fractal calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Williams Fractals on 1d high/low
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    def calculate_williams_fractals(high_arr, low_arr):
        n_days = len(high_arr)
        bearish_fractal = np.zeros(n_days, dtype=bool)
        bullish_fractal = np.zeros(n_days, dtype=bool)
        
        for i in range(2, n_days - 2):
            # Bearish fractal: high[i] is highest among high[i-2], high[i-1], high[i], high[i+1], high[i+2]
            if (high_arr[i] > high_arr[i-2] and high_arr[i] > high_arr[i-1] and 
                high_arr[i] > high_arr[i+1] and high_arr[i] > high_arr[i+2]):
                bearish_fractal[i] = True
            
            # Bullish fractal: low[i] is lowest among low[i-2], low[i-1], low[i], low[i+1], low[i+2]
            if (low_arr[i] < low_arr[i-2] and low_arr[i] < low_arr[i-1] and 
                low_arr[i] < low_arr[i+1] and low_arr[i] < low_arr[i+2]):
                bullish_fractal[i] = True
        
        return bearish_fractal, bullish_fractal
    
    bearish_fractal, bullish_fractal = calculate_williams_fractals(high_1d, low_1d)
    
    # Williams Fractals require 2 additional bars for confirmation (Williams rule)
    # Align with 2-bar delay for confirmation
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal.astype(float), additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal.astype(float), additional_delay_bars=2)
    
    # Calculate volume spike indicator (volume > 1.8 * 30-period average)
    volume_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (volume_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]):
            signals[i] = 0.0
            continue
            
        # Williams Fractal signals with volume confirmation
        bullish_fractal_signal = bullish_fractal_aligned[i] > 0.5
        bearish_fractal_signal = bearish_fractal_aligned[i] > 0.5
        vol_confirm = volume_spike[i]
        
        if position == 0:
            # Long when bullish fractal confirmed + volume spike
            if bullish_fractal_signal and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short when bearish fractal confirmed + volume spike
            elif bearish_fractal_signal and vol_confirm:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit when bearish fractal appears (trend weakness)
            if bearish_fractal_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit when bullish fractal appears (trend weakness)
            if bullish_fractal_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals