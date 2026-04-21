#!/usr/bin/env python3
"""
4h_HTF_1d_Camarilla_R1S1_Breakout_RegimeFilter_V2
Hypothesis: Combine 1d Camarilla R1/S1 breakouts with 4h choppiness regime filter to avoid false breakouts in ranging markets. 
In trending regimes (CHOP < 38.2), breakouts are more reliable. In ranging regimes (CHOP > 61.8), we fade reversals at extremes. 
Position size 0.25 for balance. Target 15-30 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')  # for 1d Camarilla levels and chop
    
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # === 1d Camarilla Pivot Levels (R1, S1) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point
    pivot = (high_1d + low_1d + close_1d) / 3.0
    # Camarilla levels
    r1 = close_1d + (high_1d - low_1d) * 1.1 / 12.0
    s1 = close_1d - (high_1d - low_1d) * 1.1 / 12.0
    
    # Align to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # === 1d Choppiness Index (CHOP) for regime filter ===
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Sum of TR over 14 periods
    sum_tr = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    # Max(high) - Min(low) over 14 periods
    max_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    range_14 = max_high - min_low
    
    # CHOP = 100 * log10(sum_tr / range_14) / log10(14)
    # Avoid division by zero and log of zero
    chop_raw = np.where(range_14 > 0, sum_tr / range_14, 1.0)
    chop_raw = np.where(chop_raw > 0, chop_raw, 1.0)
    chop = 100 * np.log10(chop_raw) / np.log10(14)
    chop = np.where(np.isnan(chop), 50.0, chop)  # neutral when undefined
    
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # === 4h Indicators ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume MA (20-period) for spike confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) 
            or np.isnan(vol_ma[i]) or np.isnan(chop_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ok = vol > 1.5 * vol_ma[i]  # volume confirmation
        chop_val = chop_aligned[i]
        
        if position == 0:
            # Regime-based entry logic
            if chop_val < 38.2:  # Trending regime - follow breakouts
                # Long: break above R1 with volume
                if price > r1_aligned[i-1] and vol_ok:
                    signals[i] = 0.25
                    position = 1
                # Short: break below S1 with volume
                elif price < s1_aligned[i-1] and vol_ok:
                    signals[i] = -0.25
                    position = -1
            elif chop_val > 61.8:  # Ranging regime - fade extremes
                # Long: reversal from S1 support with volume
                if price < s1_aligned[i-1] and vol_ok:
                    signals[i] = 0.25
                    position = 1
                # Short: reversal from R1 resistance with volume
                elif price > r1_aligned[i-1] and vol_ok:
                    signals[i] = -0.25
                    position = -1
            # In between (38.2 <= CHOP <= 61.8): no new entries, wait for clearer regime
        
        elif position == 1:
            # Exit conditions: reverse signal or volatility expansion
            if chop_val > 61.8 and price < s1_aligned[i]:  # ranging + break below S1
                signals[i] = 0.0
                position = 0
            elif price < r1_aligned[i] * 0.995:  # 0.5% below R1 as soft stop
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit conditions: reverse signal or volatility expansion
            if chop_val > 61.8 and price > r1_aligned[i]:  # ranging + break above R1
                signals[i] = 0.0
                position = 0
            elif price > s1_aligned[i] * 1.005:  # 0.5% above S1 as soft stop
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_HTF_1d_Camarilla_R1S1_Breakout_RegimeFilter_V2"
timeframe = "4h"
leverage = 1.0