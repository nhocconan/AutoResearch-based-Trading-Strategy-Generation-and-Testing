#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_CamillaPivot_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for pivot levels and ATR
    df_1d = get_htf_data(prices, '1d')
    
    # 1d daily ATR for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr_1d = np.maximum(high_1d - low_1d, 
                       np.maximum(np.abs(high_1d - np.roll(close_1d, 1)), 
                                  np.abs(low_1d - np.roll(close_1d, 1))))
    tr_1d[0] = high_1d[0] - low_1d[0]
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # 1d Camarilla pivot levels
    # Based on previous day's OHLC
    prev_high_1d = np.concatenate([[np.nan], high_1d[:-1]])
    prev_low_1d = np.concatenate([[np.nan], low_1d[:-1]])
    prev_close_1d = np.concatenate([[np.nan], close_1d[:-1]])
    
    pivot_1d = (prev_high_1d + prev_low_1d + prev_close_1d) / 3
    range_1d = prev_high_1d - prev_low_1d
    
    # Camarilla levels: R1, R2, S1, S2
    r1_1d = pivot_1d + (range_1d * 1.1 / 12)
    r2_1d = pivot_1d + (range_1d * 1.1 / 6)
    s1_1d = pivot_1d - (range_1d * 1.1 / 12)
    s2_1d = pivot_1d - (range_1d * 1.1 / 6)
    
    # Align to 4h timeframe
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    r2_1d_aligned = align_htf_to_ltf(prices, df_1d, r2_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    s2_1d_aligned = align_htf_to_ltf(prices, df_1d, s2_1d)
    
    # Volume spike filter: current volume > 2.0 x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need enough data for volume MA
    
    for i in range(start_idx, n):
        if np.isnan(pivot_1d_aligned[i]) or np.isnan(r1_1d_aligned[i]) or \
           np.isnan(r2_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or \
           np.isnan(s2_1d_aligned[i]) or np.isnan(atr_1d_aligned[i]) or \
           np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        atr = atr_1d_aligned[i]
        
        # Volume filter: require significant volume spike
        volume_ok = vol > 2.0 * vol_ma
        
        # Price position relative to Camarilla levels
        near_s1 = price <= s1_1d_aligned[i] + (0.5 * atr)  # Near support with ATR buffer
        near_s2 = price <= s2_1d_aligned[i] + (0.5 * atr)
        near_r1 = price >= r1_1d_aligned[i] - (0.5 * atr)  # Near resistance with ATR buffer
        near_r2 = price >= r2_1d_aligned[i] - (0.5 * atr)
        
        if position == 0:
            # Long setup: price near S1/S2 support with volume spike
            if (near_s1 or near_s2) and volume_ok:
                signals[i] = 0.25
                position = 1
            # Short setup: price near R1/R2 resistance with volume spike
            elif (near_r1 or near_r2) and volume_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price reaches pivot or R1, or volume dries up
            if price >= pivot_1d_aligned[i] or price >= r1_1d_aligned[i] or vol < vol_ma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price reaches pivot or S1, or volume dries up
            if price <= pivot_1d_aligned[i] or price <= s1_1d_aligned[i] or vol < vol_ma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals