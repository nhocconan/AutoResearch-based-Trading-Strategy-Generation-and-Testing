#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot levels from 1d with volume and ATR filter
# Long when price breaks above R1 with volume confirmation and ATR filter
# Short when price breaks below S1 with volume confirmation and ATR filter
# Exit when price returns to pivot point or 2x ATR stop
# Camarilla levels calculated from previous day's OHLC
# Designed to capture intraday momentum with proper risk control
# Target: 15-25 trades/year to avoid fee drag
name = "6h_Camarilla_R1_S1_Breakout_Volume_ATRFilter_v1"
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
    
    # Get 1d data for Camarilla pivot calculation (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels from previous day's OHLC
    # R4 = C + ((H-L) * 1.1/2), R3 = C + ((H-L) * 1.1/4), R2 = C + ((H-L) * 1.1/6), R1 = C + ((H-L) * 1.1/12)
    # S1 = C - ((H-L) * 1.1/12), S2 = C - ((H-L) * 1.1/6), S3 = C - ((H-L) * 1.1/4), S4 = C - ((H-L) * 1.1/2)
    H_1d = df_1d['high'].values
    L_1d = df_1d['low'].values
    C_1d = df_1d['close'].values
    
    # Calculate Camarilla levels
    range_1d = H_1d - L_1d
    R1 = C_1d + (range_1d * 1.1 / 12)
    S1 = C_1d - (range_1d * 1.1 / 12)
    PP = (H_1d + L_1d + C_1d) / 3  # Pivot point
    
    # Align Camarilla levels to 6h timeframe (previous day's levels available at open)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    PP_aligned = align_htf_to_ltf(prices, df_1d, PP)
    
    # 6h ATR for stops
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr_6h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        if np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or np.isnan(PP_aligned[i]) or np.isnan(atr_6h[i]):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x average volume (20-period)
        if i >= 20:
            avg_volume = np.mean(volume[i-20:i])
        else:
            avg_volume = volume[i]
        volume_filter = volume[i] > 1.5 * avg_volume
        
        if position == 0:
            # Long: Price breaks above R1 with volume confirmation
            if close[i] > R1_aligned[i] and volume_filter:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S1 with volume confirmation
            elif close[i] < S1_aligned[i] and volume_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Price returns to pivot point or 2x ATR stop
            if close[i] < PP_aligned[i] or close[i] < close[i-1] - 2.0 * atr_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Price returns to pivot point or 2x ATR stop
            if close[i] > PP_aligned[i] or close[i] > close[i-1] + 2.0 * atr_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals