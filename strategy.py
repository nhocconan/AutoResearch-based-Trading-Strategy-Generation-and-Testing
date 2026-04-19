#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Camarilla_R1S1_1dVolumeSpike_CloseAboveVWAP"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for Camarilla and VWAP
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels (R1, S1) from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point and Camarilla levels
    pivot = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    r1 = pivot + (range_1d * 1.1 / 12)
    s1 = pivot - (range_1d * 1.1 / 12)
    
    # Calculate VWAP for 4h data
    typical_price = (prices['high'] + prices['low'] + prices['close']) / 3
    vwap = (typical_price * prices['volume']).cumsum() / prices['volume'].cumsum()
    vwap_values = vwap.values
    
    # Volume spike detection (volume > 2.0 * 20-period average)
    volume_ma = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    volume_spike = prices['volume'].values > (volume_ma * 2.0)
    
    # Align Camarilla levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(vwap_values[i]):
            signals[i] = 0.0
            continue
            
        price = prices['close'].iloc[i]
        vol_confirm = volume_spike[i]
        price_above_vwap = price > vwap_values[i]
        
        if position == 0:
            # Long when price crosses above R1 with volume spike and above VWAP
            if price > r1_aligned[i] and vol_confirm and price_above_vwap:
                signals[i] = 0.25
                position = 1
            # Short when price crosses below S1 with volume spike and below VWAP
            elif price < s1_aligned[i] and vol_confirm and not price_above_vwap:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit when price crosses below S1 or loses VWAP
            if price < s1_aligned[i] or price < vwap_values[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit when price crosses above R1 or regains VWAP
            if price > r1_aligned[i] or price > vwap_values[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals