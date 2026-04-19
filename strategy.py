#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_Camarilla_R1S1_Breakout_Volume_V1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels for each 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    def calculate_camarilla(high_arr, low_arr, close_arr):
        n_days = len(high_arr)
        R1 = np.full(n_days, np.nan)
        S1 = np.full(n_days, np.nan)
        
        for i in range(n_days):
            high_val = high_arr[i]
            low_val = low_arr[i]
            close_val = close_arr[i]
            
            range_val = high_val - low_val
            R1[i] = close_val + range_val * 1.1 / 12
            S1[i] = close_val - range_val * 1.1 / 12
        
        return R1, S1
    
    R1_1d, S1_1d = calculate_camarilla(high_1d, low_1d, close_1d)
    
    # Calculate volume spike indicator (volume > 1.5 * 20-period average)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 1.5)
    
    # Align Camarilla levels to 12h timeframe
    R1_12h = align_htf_to_ltf(prices, df_1d, R1_1d)
    S1_12h = align_htf_to_ltf(prices, df_1d, S1_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(R1_12h[i]) or np.isnan(S1_12h[i]):
            signals[i] = 0.0
            continue
            
        # Camarilla breakout signals with volume confirmation
        vol_confirm = volume_spike[i]
        
        if position == 0:
            # Long when price breaks above R1 with volume spike
            if close[i] > R1_12h[i] and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below S1 with volume spike
            elif close[i] < S1_12h[i] and vol_confirm:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit when price falls below S1 (reversal signal)
            if close[i] < S1_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit when price rises above R1 (reversal signal)
            if close[i] > R1_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals