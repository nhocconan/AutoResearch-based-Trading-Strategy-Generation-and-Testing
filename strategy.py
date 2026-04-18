#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h/1d pivot reversal with volume confirmation and ATR stop.
# Long when price crosses above S1 with volume > 2x 24-period average and ATR > 0.
# Short when price crosses below R1 with same conditions.
# Exit when price crosses back to pivot (P).
# Uses 1d pivot levels for mean-reversion, volume surge for conviction, ATR for volatility.
# Designed for ~20-40 trades/year per symbol.
name = "4h_1dPivot_S1R1_Reversal_Volume_ATR"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for pivot levels
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d pivot points: P = (H+L+C)/3, S1 = 2*P - H, R1 = 2*P - L
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    s1_1d = 2 * pivot_1d - high_1d
    r1_1d = 2 * pivot_1d - low_1d
    
    # Align pivot levels to 4h timeframe
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    
    # ATR(14) on 1d for volatility filter
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - close_1d)
    tr3 = np.abs(low_1d - close_1d)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Volume filter: current volume > 2.0 * 24-period average (24 * 4h = 4 days)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_filter = volume > (2.0 * vol_ma_24)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pivot_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or 
            np.isnan(r1_1d_aligned[i]) or np.isnan(atr_1d_aligned[i]) or
            np.isnan(vol_ma_24[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        pivot_val = pivot_1d_aligned[i]
        s1_val = s1_1d_aligned[i]
        r1_val = r1_1d_aligned[i]
        atr_val = atr_1d_aligned[i]
        vol_filter = volume_filter[i]
        
        if position == 0:
            # Long: price crosses above S1 with volume surge and volatility
            if close_val > s1_val and close_val <= pivot_val and vol_filter and atr_val > 0:
                signals[i] = 0.25
                position = 1
            # Short: price crosses below R1 with volume surge and volatility
            elif close_val < r1_val and close_val >= pivot_val and vol_filter and atr_val > 0:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses back to pivot
            if close_val >= pivot_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses back to pivot
            if close_val <= pivot_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals