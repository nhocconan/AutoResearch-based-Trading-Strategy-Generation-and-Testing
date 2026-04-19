#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h timeframe with 1d pivot (R1/S1) breakout, volume confirmation, and ATR volatility filter.
# Uses daily pivot levels for structure, trades breakouts with volume confirmation in trending markets.
# Designed to work in both bull and bear markets by filtering trades with volatility regime (ATR-based).
# Target: 15-30 trades/year to minimize fee drag and avoid overtrading.

name = "12h_Pivot_R1S1_Breakout_VolumeATR_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for pivot and ATR calculation (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Daily high, low, close for pivot calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot point and support/resistance levels
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    r1_1d = close_1d + range_1d * 1.1 / 12  # R1 = C + (H-L)*1.1/12
    s1_1d = close_1d - range_1d * 1.1 / 12  # S1 = C - (H-L)*1.1/12
    
    # Align daily pivot levels to 12h timeframe (waits for daily close)
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Daily ATR (14-period) for volatility filter
    tr = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
    tr = np.maximum(tr, np.abs(low_1d[1:] - close_1d[:-1]))
    tr = np.concatenate([[np.nan], tr])
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period average (12h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(pivot_1d_aligned[i]) or np.isnan(r1_1d_aligned[i]) or 
            np.isnan(s1_1d_aligned[i]) or np.isnan(atr_14_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        pivot = pivot_1d_aligned[i]
        r1 = r1_1d_aligned[i]
        s1 = s1_1d_aligned[i]
        atr = atr_14_1d_aligned[i]
        
        volume_confirmed = vol > 1.5 * vol_ma
        volatility_filter = atr > 0  # Ensure valid ATR
        
        if position == 0:
            # Long: break above R1 with volume and volatility
            if price > r1 and volume_confirmed and volatility_filter:
                signals[i] = 0.25
                position = 1
            # Short: break below S1 with volume and volatility
            elif price < s1 and volume_confirmed and volatility_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price below pivot
            if price < pivot:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price above pivot
            if price > pivot:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals