#!/usr/bin/env python3
"""
4h_Pivot_Reversion_Volume_Trend
Hypothesis: In range-bound markets (Chop > 61.8), price reverts to daily pivot (S2/R2) with volume confirmation. In trending markets (Chop < 38.2), price follows 1d EMA50 trend. Combines mean reversion and trend following with regime filter to work in both bull and bear markets. Target: 20-40 trades/year.
"""

name = "4h_Pivot_Reversion_Volume_Trend"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for pivots and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot points (standard formula)
    pivot = (high_1d + low_1d + close_1d) / 3.0
    r1 = 2 * pivot - low_1d
    s1 = 2 * pivot - high_1d
    r2 = pivot + (high_1d - low_1d)
    s2 = pivot - (high_1d - low_1d)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Choppiness Index (14-period) on 1d for regime filter
    atr_1d = pd.Series(np.maximum(
        high_1d - low_1d,
        np.maximum(
            np.abs(high_1d - np.concatenate([[high_1d[0]], high_1d[:-1]])),
            np.abs(low_1d - np.concatenate([[low_1d[0]], low_1d[:-1]]))
        )
    )).rolling(window=14, min_periods=14).mean()
    
    true_range_sum = atr_1d * 14
    max_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max()
    min_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min()
    chop = 100 * np.log10(true_range_sum / (max_high - min_low)) / np.log10(14)
    chop = chop.values
    
    # Align 1d indicators to 4h timeframe
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Volume average (20-period) for volume confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(s2_aligned[i]) or np.isnan(r2_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(chop_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume[i] > 1.5 * vol_ma_20[i]
        
        chop_val = chop_aligned[i]
        price = close[i]
        
        if position == 0:
            # Range market (Chop > 61.8): mean reversion to S2/R2
            if chop_val > 61.8:
                # Long near S2 with volume confirmation
                if price <= s2_aligned[i] * 1.005 and vol_confirm:
                    signals[i] = 0.25
                    position = 1
                # Short near R2 with volume confirmation
                elif price >= r2_aligned[i] * 0.995 and vol_confirm:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            # Trending market (Chop < 38.2): follow 1d EMA50 trend
            elif chop_val < 38.2:
                # Long in uptrend
                if price > ema_50_1d_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short in downtrend
                elif price < ema_50_1d_aligned[i]:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            # Neutral chop: stay flat
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long conditions
            if chop_val > 61.8 and price >= pivot[i] * 0.995:  # Return to pivot in range
                signals[i] = 0.0
                position = 0
            elif chop_val < 38.2 and price < ema_50_1d_aligned[i]:  # Trend reversal
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short conditions
            if chop_val > 61.8 and price <= pivot[i] * 1.005:  # Return to pivot in range
                signals[i] = 0.0
                position = 0
            elif chop_val < 38.2 and price > ema_50_1d_aligned[i]:  # Trend reversal
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals