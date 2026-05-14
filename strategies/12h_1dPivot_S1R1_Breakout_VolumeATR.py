#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1dPivot_S1R1_Breakout_VolumeATR"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for pivot points and ATR calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily ATR(14) on daily data
    tr1 = np.maximum(high_1d[1:], close_1d[:-1]) - np.minimum(low_1d[1:], close_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate daily pivot points: P = (H+L+C)/3
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    # Support and resistance levels
    s1_1d = 2 * pivot_1d - high_1d
    r1_1d = 2 * pivot_1d - low_1d
    
    # Align pivot levels to 12h timeframe
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need volume MA and ATR data
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pivot_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or 
            np.isnan(r1_1d_aligned[i]) or np.isnan(atr_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        atr = atr_1d_aligned[i]
        
        # Volume and volatility filters
        volume_confirmed = vol > 1.5 * vol_ma
        
        # Pivot levels
        s1 = s1_1d_aligned[i]
        r1 = r1_1d_aligned[i]
        pivot = pivot_1d_aligned[i]
        
        if position == 0:
            # Long: Breakout above R1 with volume confirmation
            if price > r1 and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: Breakdown below S1 with volume confirmation
            elif price < s1 and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price closes below S1 (mean reversion) or ATR-based stop
            if price < s1 or price < (high[i] - 2.0 * atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price closes above R1 (mean reversion) or ATR-based stop
            if price > r1 or price > (low[i] + 2.0 * atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals