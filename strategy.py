#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_Camarilla_R1S1_Breakout_Volume_ATRFilter_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla and ATR
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels for previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's range
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    
    # Handle first value
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    
    # Camarilla levels
    R1 = prev_close + 1.1 * (prev_high - prev_low) / 12
    S1 = prev_close - 1.1 * (prev_high - prev_low) / 12
    R2 = prev_close + 1.1 * (prev_high - prev_low) / 6
    S2 = prev_close - 1.1 * (prev_high - prev_low) / 6
    R3 = prev_close + 1.1 * (prev_high - prev_low) / 4
    S3 = prev_close - 1.1 * (prev_high - prev_low) / 4
    R4 = prev_close + 1.1 * (prev_high - prev_low) / 2
    S4 = prev_close - 1.1 * (prev_high - prev_low) / 2
    
    # ATR for volatility filter
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = high_1d[0] - low_1d[0]  # first TR
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align to 6h timeframe
    R1_6h = align_htf_to_ltf(prices, df_1d, R1)
    S1_6h = align_htf_to_ltf(prices, df_1d, S1)
    R4_6h = align_htf_to_ltf(prices, df_1d, R4)
    S4_6h = align_htf_to_ltf(prices, df_1d, S4)
    atr_6h = align_htf_to_ltf(prices, df_1d, atr)
    
    # Volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(R1_6h[i]) or np.isnan(S1_6h[i]) or 
            np.isnan(R4_6h[i]) or np.isnan(S4_6h[i]) or
            np.isnan(atr_6h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
            
        # Volume filter: current volume > 1.5x average
        volume_filter = volume[i] > 1.5 * vol_ma[i]
        
        # ATR filter: avoid extremely low volatility
        atr_filter = atr_6h[i] > 0
        
        if position == 0:
            # Long breakout above R1 with volume
            if (close[i] > R1_6h[i] and 
                volume_filter and 
                atr_filter):
                signals[i] = 0.25
                position = 1
            # Short breakdown below S1 with volume
            elif (close[i] < S1_6h[i] and 
                  volume_filter and 
                  atr_filter):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit at R4 (take profit) or if price falls back below R1
            if (close[i] >= R4_6h[i] or 
                close[i] < R1_6h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit at S4 (take profit) or if price rises back above S1
            if (close[i] <= S4_6h[i] or 
                close[i] > S1_6h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals