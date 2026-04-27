#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Aroon oscillator with 1d volume-weighted average price (VWAP) trend filter and volume confirmation
# Uses Aroon oscillator (25-period) to detect trend strength and direction on 6h timeframe.
# Trend filter: 1d VWAP - price above VWAP indicates uptrend, below indicates downtrend.
# Entry: Aroon oscillator crossing above +50 for long, below -50 for short with volume confirmation (1.5x 20-period average volume).
# Exit: Aroon oscillator crossing back through zero or trend reversal.
# Works in both bull and bear markets by combining momentum (Aroon) with trend (VWAP) and volume confirmation.
# Target: 20-30 trades/year to minimize fee decay while capturing sustained moves.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for VWAP trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Aroon oscillator (25-period) on 6h
    aroon_period = 25
    aroon_up = np.full(n, np.nan)
    aroon_down = np.full(n, np.nan)
    aroon_osc = np.full(n, np.nan)
    
    for i in range(aroon_period, n):
        # Find periods since highest high and lowest low
        highest_high_idx = i - np.argmax(high[i-aroon_period:i+1])
        lowest_low_idx = i - np.argmin(low[i-aroon_period:i+1])
        
        aroon_up[i] = ((aroon_period - (i - highest_high_idx)) / aroon_period) * 100
        aroon_down[i] = ((aroon_period - (i - lowest_low_idx)) / aroon_period) * 100
        aroon_osc[i] = aroon_up[i] - aroon_down[i]
    
    # Calculate 1d VWAP
    typical_price_1d = (df_1d['high'].values + df_1d['low'].values + df_1d['close'].values) / 3
    vwap_num = np.cumsum(typical_price_1d * df_1d['volume'].values)
    vwap_den = np.cumsum(df_1d['volume'].values)
    vwap_1d = vwap_num / vwap_den
    
    # Calculate 20-period average volume on 6h for spike detection
    vol_ma = np.full(n, np.nan)
    vol_period = 20
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    # Align 1d indicators to 6h timeframe
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    signals = np.zeros(n)
    position = 0
    size = 0.25
    
    # Warmup period
    start_idx = max(aroon_period, vol_period) + 1
    
    for i in range(start_idx, n):
        if (np.isnan(aroon_osc[i]) or 
            np.isnan(vwap_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Volume confirmation: at least 1.5x average volume
        volume_confirmation = vol_ratio > 1.5
        
        if position == 0:
            # Long: Aroon oscillator crosses above +50 with uptrend and volume
            if aroon_osc[i] > 50 and aroon_osc[i-1] <= 50 and price > vwap_1d_aligned[i] and volume_confirmation:
                signals[i] = size
                position = 1
            # Short: Aroon oscillator crosses below -50 with downtrend and volume
            elif aroon_osc[i] < -50 and aroon_osc[i-1] >= -50 and price < vwap_1d_aligned[i] and volume_confirmation:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: Aroon oscillator crosses below zero or trend reversal
            if aroon_osc[i] < 0 or price < vwap_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: Aroon oscillator crosses above zero or trend reversal
            if aroon_osc[i] > 0 or price > vwap_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_AroonOscillator_1dVWAP_Volume"
timeframe = "6h"
leverage = 1.0