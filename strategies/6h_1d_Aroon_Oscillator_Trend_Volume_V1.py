#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_Aroon_Oscillator_Trend_Volume_V1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Aroon calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Aroon Oscillator on 1d high/low
    period = 25
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    def calculate_aroon(high_arr, low_arr, period):
        n_days = len(high_arr)
        aroon_up = np.full(n_days, np.nan)
        aroon_down = np.full(n_days, np.nan)
        
        for i in range(period - 1, n_days):
            # Periods since highest high
            highest_high_idx = np.argmax(high_arr[i - period + 1:i + 1]) + i - period + 1
            periods_since_high = i - highest_high_idx
            aroon_up[i] = ((period - periods_since_high) / period) * 100
            
            # Periods since lowest low
            lowest_low_idx = np.argmin(low_arr[i - period + 1:i + 1]) + i - period + 1
            periods_since_low = i - lowest_low_idx
            aroon_down[i] = ((period - periods_since_low) / period) * 100
        
        aroon_osc = aroon_up - aroon_down
        return aroon_osc
    
    aroon_osc = calculate_aroon(high_1d, low_1d, period)
    
    # Calculate volume spike indicator (volume > 1.5 * 20-period average)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 1.5)
    
    # Align Aroon Oscillator to 6h timeframe
    aroon_osc_aligned = align_htf_to_ltf(prices, df_1d, aroon_osc)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(aroon_osc_aligned[i]):
            signals[i] = 0.0
            continue
            
        # Aroon Oscillator signals:
        # > 50: strong uptrend, < -50: strong downtrend
        # Volume confirmation required
        vol_confirm = volume_spike[i]
        
        if position == 0:
            # Long when strong uptrend + volume spike
            if aroon_osc_aligned[i] > 50 and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short when strong downtrend + volume spike
            elif aroon_osc_aligned[i] < -50 and vol_confirm:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit when trend weakens or reverses
            if aroon_osc_aligned[i] < 0:  # Trend turning negative
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit when trend weakens or reverses
            if aroon_osc_aligned[i] > 0:  # Trend turning positive
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals