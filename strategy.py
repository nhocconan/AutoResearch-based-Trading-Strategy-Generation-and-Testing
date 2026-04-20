#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_Cam_Pivot_R1S1_SwingReject"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily pivot points
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot = (H + L + C) / 3
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    # R1 = 2*P - L, S1 = 2*P - H
    r1_1d = 2 * pivot_1d - low_1d
    s1_1d = 2 * pivot_1d - high_1d
    # R2 = P + (H - L), S2 = P - (H - L)
    r2_1d = pivot_1d + (high_1d - low_1d)
    s2_1d = pivot_1d - (high_1d - low_1d)
    # R3 = H + 2*(P - L), S3 = L - 2*(H - P)
    r3_1d = high_1d + 2 * (pivot_1d - low_1d)
    s3_1d = low_1d - 2 * (high_1d - pivot_1d)
    
    # Align pivot levels to 12h timeframe
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    r2_1d_aligned = align_htf_to_ltf(prices, df_1d, r2_1d)
    s2_1d_aligned = align_htf_to_ltf(prices, df_1d, s2_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    # Calculate 12h ATR(14) for volatility filtering
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 12h volume average for confirmation
    vol_ma = pd.Series(prices['volume'].values).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 30  # Need enough data for ATR and volume MA
    
    for i in range(start_idx, n):
        # Get aligned values
        pivot = pivot_1d_aligned[i]
        r1 = r1_1d_aligned[i]
        s1 = s1_1d_aligned[i]
        r2 = r2_1d_aligned[i]
        s2 = s2_1d_aligned[i]
        r3 = r3_1d_aligned[i]
        s3 = s3_1d_aligned[i]
        current_atr = atr[i]
        current_vol = prices['volume'].iloc[i]
        current_vol_ma = vol_ma[i]
        current_close = prices['close'].iloc[i]
        
        # Skip if any value is NaN
        if (np.isnan(pivot) or np.isnan(r1) or np.isnan(s1) or
            np.isnan(r2) or np.isnan(s2) or np.isnan(r3) or np.isnan(s3) or
            np.isnan(current_atr) or np.isnan(current_vol_ma)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: current volume > 1.5x 24-period average
        vol_condition = current_vol > 1.5 * current_vol_ma
        
        if position == 0:
            # Long: Price rejects S1 support with volume (bounce)
            if (current_close > s1 and
                current_close < pivot and
                vol_condition and
                prices['close'].iloc[i-1] <= s1):
                signals[i] = 0.25
                position = 1
                entry_price = current_close
            
            # Short: Price rejects R1 resistance with volume (rejection)
            elif (current_close < r1 and
                  current_close > pivot and
                  vol_condition and
                  prices['close'].iloc[i-1] >= r1):
                signals[i] = -0.25
                position = -1
                entry_price = current_close
        
        elif position == 1:
            # Long exit: price breaks below S1 or reaches R1
            if current_close <= s1 or current_close >= r1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above R1 or reaches S1
            if current_close >= r1 or current_close <= s1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals