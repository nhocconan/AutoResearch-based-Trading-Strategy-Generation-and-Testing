#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d volume spike and chop regime filter.
Long when price breaks above Donchian upper (20-period high) AND 1d volume > 1.5x 20-period average 
AND 1d chop > 61.8 (ranging market - mean reversion setup).
Short when price breaks below Donchian lower (20-period low) AND 1d volume > 1.5x 20-period average 
AND 1d chop > 61.8 (ranging market - mean reversion setup).
Exit on opposite Donchian break or when chop < 38.2 (trending market - exit ranges).
Uses 1d for volume/chop filters to avoid lower-timeframe noise, 12h for Donchian breakouts.
Target: 50-150 total trades over 4 years (12-37/year). Donchian captures breakouts, 
volume confirms conviction, chop filter ensures we trade ranges where mean reversion works.
In bear markets like 2025, ranging behavior increases, making this strategy more relevant.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
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
    
    # Calculate 1d volume SMA(20)
    vol_1d_series = pd.Series(vol_1d)
    vol_sma_1d = vol_1d_series.rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d Chopiness Index(14)
    # CHOP = 100 * log10(sum(ATR(1),14) / (log10(HH(14)-LL(14)) * sqrt(14)))
    tr1 = np.maximum(high_1d[1:] - low_1d[:-1], 
                     np.maximum(np.abs(high_1d[1:] - close_1d[:-1]),
                                np.abs(low_1d[1:] - close_1d[:-1])))
    tr1 = np.concatenate([[np.nan], tr1])  # align with index 0
    atr1 = pd.Series(tr1).rolling(window=1, min_periods=1).mean().values  # ATR(1) = TR
    sum_atr1 = pd.Series(atr1).rolling(window=14, min_periods=14).sum().values
    hh_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    range_14 = hh_14 - ll_14
    chop_1d = 100 * np.log10(sum_atr1 / (np.log10(range_14) * np.sqrt(14)))
    
    # Align 1d filters to 12h timeframe
    vol_sma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_sma_1d)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Calculate 12h Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donch_hi = high_series.rolling(window=20, min_periods=20).max().values
    donch_lo = low_series.rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 60  # warmup for indicators (max of 20, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(vol_sma_1d_aligned[i]) or np.isnan(chop_1d_aligned[i]) or 
            np.isnan(donch_hi[i]) or np.isnan(donch_lo[i])):
            signals[i] = 0.0
            continue
        
        vol_1d_now = vol_1d[i // 16] if i // 16 < len(vol_1d) else vol_1d[-1]  # approximate 12h->1d volume
        vol_avg = vol_sma_1d_aligned[i]
        chop = chop_1d_aligned[i]
        price = close[i]
        upper = donch_hi[i]
        lower = donch_lo[i]
        
        if position == 0:
            # Long: price > Donchian upper AND volume spike AND chop > 61.8 (range)
            if price > upper and vol_1d_now > 1.5 * vol_avg and chop > 61.8:
                signals[i] = 0.25
                position = 1
            # Short: price < Donchian lower AND volume spike AND chop > 61.8 (range)
            elif price < lower and vol_1d_now > 1.5 * vol_avg and chop > 61.8:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price < Donchian lower OR chop < 38.2 (trend)
            if price < lower or chop < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price > Donchian upper OR chop < 38.2 (trend)
            if price > upper or chop < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_VolumeSpike_ChopFilter"
timeframe = "12h"
leverage = 1.0