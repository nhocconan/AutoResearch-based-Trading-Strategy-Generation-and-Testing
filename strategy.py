#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout + 1d volume spike + choppiness regime filter.
Long when price breaks above 20-bar Donchian high AND 1d volume > 1.5x 20-bar average AND chop > 61.8 (range market).
Short when price breaks below 20-bar Donchian low AND 1d volume > 1.5x 20-bar average AND chop > 61.8.
Exit on Donchian opposite breakout or chop < 38.2 (trending market).
Uses 1d for volume and chop filters, 12h for Donchian calculation.
Target: 50-150 total trades over 4 years (12-37/year). Donchian breakouts capture trends, 
volume confirms conviction, chop filter avoids whipsaws in strong trends.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for volume and chop filters
    df_1d = get_htf_data(prices, '1d')
    vol_1d = df_1d['volume'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d volume 20-bar average
    vol_1d_series = pd.Series(vol_1d)
    vol_ma20_1d = vol_1d_series.rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d choppiness index (CHOP)
    # CHOP = 100 * log10(sum(ATR(1)) / (n * (HHV - LLV))) / log10(n)
    # Simplified: CHOP = 100 * log10(ATR_sum / (n * (max_high - min_low))) / log10(n)
    tr1 = np.maximum(high_1d - low_1d, np.absolute(high_1d - np.roll(close_1d, 1)), np.absolute(low_1d - np.roll(close_1d, 1)))
    tr1[0] = high_1d[0] - low_1d[0]  # first TR
    atr1 = tr1  # ATR(1) is just TR
    
    # Sum of ATR(1) over 14 periods
    atr_sum = pd.Series(atr1).rolling(window=14, min_periods=14).sum().values
    hhvl = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values - pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    # Avoid division by zero
    chop_raw = 100 * np.log10(atr_sum / (14 * hhvl + 1e-10)) / np.log10(14)
    chop = np.where(hhvl > 0, chop_raw, 50.0)  # default to 50 when no range
    
    # Align 1d volume MA and chop to 12h timeframe
    vol_ma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma20_1d)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Calculate 12h Donchian channels (20-bar)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donch_high = high_series.rolling(window=20, min_periods=20).max().values
    donch_low = low_series.rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if np.isnan(vol_ma20_1d_aligned[i]) or np.isnan(chop_aligned[i]) or np.isnan(donch_high[i]) or np.isnan(donch_low[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ma = vol_ma20_1d_aligned[i]
        vol_current = vol_1d[i // 12] if i // 12 < len(vol_1d) else vol_1d[-1]  # approximate 12h to 1d volume
        chop_val = chop_aligned[i]
        upper = donch_high[i]
        lower = donch_low[i]
        
        if position == 0:
            # Long: price > Donchian high AND volume spike AND chop > 61.8 (range)
            if price > upper and vol_current > 1.5 * vol_ma and chop_val > 61.8:
                signals[i] = 0.25
                position = 1
            # Short: price < Donchian low AND volume spike AND chop > 61.8 (range)
            elif price < lower and vol_current > 1.5 * vol_ma and chop_val > 61.8:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price < Donchian low OR chop < 38.2 (trending)
            if price < lower or chop_val < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price > Donchian high OR chop < 38.2 (trending)
            if price > upper or chop_val < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_VolumeSpike_ChopFilter"
timeframe = "12h"
leverage = 1.0