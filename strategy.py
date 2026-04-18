#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h mean reversion at daily pivot S1/R1 with volume confirmation and ATR volatility filter.
# Go long when price touches or breaks below S1 with volume > 1.5x 20-period average and ATR > 0.
# Go short when price touches or breaks above R1 with same conditions.
# Exit when price crosses back to the daily pivot point.
# Uses daily pivot levels for mean reversion zones, volume surge for conviction, ATR for volatility filter.
# Designed for ~15-30 trades/year per symbol, works in both trending and ranging markets.
name = "6h_Pivot_S1R1_MeanReversion_Volume_ATR"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for pivot calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily pivot points: P = (H + L + C)/3, S1 = 2P - H, R1 = 2P - L
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    s1_1d = 2 * pivot_1d - high_1d
    r1_1d = 2 * pivot_1d - low_1d
    
    # Align pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    
    # ATR(14) on 6h for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.concatenate([[close[0]], close[:-1]]))  # |H - Cprev|
    tr3 = np.abs(low - np.concatenate([[close[0]], close[:-1]]))   # |L - Cprev|
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume filter: current volume > 1.5 * 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pivot_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        pivot_val = pivot_aligned[i]
        s1_val = s1_aligned[i]
        r1_val = r1_aligned[i]
        atr_val = atr[i]
        vol_filter = volume_filter[i]
        
        if position == 0:
            # Long: price at or below S1 with volume surge and volatility
            if close_val <= s1_val and vol_filter and atr_val > 0:
                signals[i] = 0.25
                position = 1
            # Short: price at or above R1 with volume surge and volatility
            elif close_val >= r1_val and vol_filter and atr_val > 0:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses back to pivot (mean reversion complete)
            if close_val >= pivot_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses back to pivot (mean reversion complete)
            if close_val <= pivot_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals