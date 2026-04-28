#!/usr/bin/env python3
"""
6h_TRIX_VolumeSpike_Regime
Hypothesis: On 6-hour timeframe, use TRIX (15-period) momentum with volume spike confirmation and Choppiness Index regime filter. TRIX captures smoothed momentum, volume surge confirms institutional participation, and Choppiness Index avoids whipsaws in ranging markets. Designed for low trade frequency (~20-40/year) with high win rate in both bull and bear markets by focusing on strong momentum bursts in trending regimes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for TRIX calculation (smoother on higher TF)
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 30:
        return np.zeros(n)
    
    # Calculate TRIX (15-period) on daily close
    close_daily = df_daily['close'].values
    # EMA1
    ema1 = pd.Series(close_daily).ewm(span=15, adjust=False, min_periods=15).mean().values
    # EMA2 of EMA1
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    # EMA3 of EMA2
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    # TRIX = (EMA3 - previous EMA3) / previous EMA3 * 100
    trix_raw = np.zeros_like(ema3)
    trix_raw[1:] = (ema3[1:] - ema3[:-1]) / ema3[:-1] * 100
    # Smooth TRIX with 9-period EMA (signal line)
    trix = pd.Series(trix_raw).ewm(span=9, adjust=False, min_periods=9).mean().values
    
    # Align daily TRIX to 6h timeframe
    trix_aligned = align_htf_to_ltf(prices, df_daily, trix)
    
    # Get 12h data for Choppiness Index (regime filter)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate Choppiness Index (14-period) on 12h data
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # ATR (14-period)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Sum of TR over 14 periods
    sum_tr = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index = 100 * log10(sum TR / (HH - LL)) / log10(14)
    # Avoid division by zero
    hl_range = hh - ll
    chop_raw = np.zeros_like(sum_tr)
    mask = (hl_range > 0) & ~np.isnan(hl_range)
    chop_raw[mask] = 100 * np.log10(sum_tr[mask] / hl_range[mask]) / np.log10(14)
    
    # Align Chop to 6h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_12h, chop_raw)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_surge = volume > (vol_ma_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(trix_aligned[i]) or np.isnan(chop_aligned[i]) or
            np.isnan(volume_surge[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: Choppiness Index < 50 = trending (avoid ranging markets)
        is_trending = chop_aligned[i] < 50.0
        
        # Entry conditions: TRIX momentum + volume surge + trending regime
        long_entry = (trix_aligned[i] > 0.1) and volume_surge[i] and is_trending
        short_entry = (trix_aligned[i] < -0.1) and volume_surge[i] and is_trending
        
        # Exit when TRIX reverses or volume drops
        long_exit = (trix_aligned[i] < -0.05) or not volume_surge[i]
        short_exit = (trix_aligned[i] > 0.05) or not volume_surge[i]
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0  # Exit to flat
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0  # Exit to flat
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_TRIX_VolumeSpike_Regime"
timeframe = "6h"
leverage = 1.0