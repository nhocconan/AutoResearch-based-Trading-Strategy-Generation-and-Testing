#!/usr/bin/env python3
"""
12h Camarilla Pivot + Volume Spike + Choppiness Regime Filter
Long when price touches Camarilla S3 support with volume spike in choppy market
Short when price touches Camarilla R3 resistance with volume spike in choppy market
Exit when price crosses midline (Pivot) or opposite touch
Designed for range-bound markets with mean reversion at extreme levels
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d data for Camarilla pivot calculation ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for previous day
    # R4 = C + ((H-L)*1.1/2)
    # R3 = C + ((H-L)*1.1/4)
    # R2 = C + ((H-L)*1.1/6)
    # R1 = C + ((H-L)*1.1/12)
    # PP = (H+L+C)/3
    # S1 = C - ((H-L)*1.1/12)
    # S2 = C - ((H-L)*1.1/6)
    # S3 = C - ((H-L)*1.1/4)
    # S4 = C - ((H-L)*1.1/2)
    
    hl_range = high_1d - low_1d
    camarilla_pp = (high_1d + low_1d + close_1d) / 3
    camarilla_r3 = close_1d + (hl_range * 1.1 / 4)
    camarilla_s3 = close_1d - (hl_range * 1.1 / 4)
    
    # Align to 12h timeframe
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # === Volume confirmation (20-period volume average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / (vol_ma + 1e-10)
    
    # === Choppiness Index regime filter (14-period) ===
    # CHOP = 100 * log10(sum(TR)/ (HHV(high) - LLV(low))) / log10(14)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / (hh - ll + 1e-10)) / np.log10(14)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        if (np.isnan(camarilla_pp_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(vol_ratio[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses above pivot (mean reversion complete) or touches R3
            if close[i] > camarilla_pp_aligned[i] or close[i] >= camarilla_r3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses below pivot or touches S3
            if close[i] < camarilla_pp_aligned[i] or close[i] <= camarilla_s3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Need volume spike (above average)
            if vol_ratio[i] < 1.5:
                signals[i] = 0.0
                continue
            
            # Only trade in choppy/range-bound markets (Choppiness > 50)
            if chop[i] < 50:
                signals[i] = 0.0
                continue
            
            # Entry: Price touches Camarilla extreme levels with volume spike
            if close[i] <= camarilla_s3_aligned[i]:
                # Touch S3 support -> long (mean reversion long)
                position = 1
                signals[i] = 0.25
            elif close[i] >= camarilla_r3_aligned[i]:
                # Touch R3 resistance -> short (mean reversion short)
                position = -1
                signals[i] = -0.25
    
    return signals