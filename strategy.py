#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Camarilla_R1_S1_Breakout_Volume_ATRFilter"
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
    
    # Get daily data for Camarilla pivot points
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate previous day's Camarilla levels (R1, S1)
    prev_high = np.concatenate([[np.nan], high_1d[:-1]])
    prev_low = np.concatenate([[np.nan], low_1d[:-1]])
    prev_close = np.concatenate([[np.nan], close_1d[:-1]])
    
    pivot = (prev_high + prev_low + prev_close) / 3
    r1 = pivot + (prev_high - prev_low) * 1.0833
    s1 = pivot - (prev_high - prev_low) * 1.0833
    
    # Align daily Camarilla levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR filter: ATR(14) > 0.02 * price (minimum volatility)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first value
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20, 14)  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        if np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(vol_ma_20[i]) or np.isnan(atr[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        atr_val = atr[i]
        
        # Volume and volatility filters
        volume_ok = vol > 1.5 * vol_ma
        volatility_ok = atr_val > 0.02 * price
        
        if position == 0:
            # Long: price breaks above R1 with volume and volatility
            if price > r1_aligned[i] and volume_ok and volatility_ok:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume and volatility
            elif price < s1_aligned[i] and volume_ok and volatility_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price returns below S1 (mean reversion to opposite level)
            if price < s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price returns above R1 (mean reversion to opposite level)
            if price > r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals