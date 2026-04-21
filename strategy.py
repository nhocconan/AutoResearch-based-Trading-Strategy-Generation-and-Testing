#!/usr/bin/env python3
"""
1d_WeeklyDonchian_Breakout_VolumeRegime_V1
Hypothesis: Use 1w Donchian(20) breakout as primary signal on 1d timeframe, with volume confirmation (>1.5x 20d MA) and chop regime filter (CHOP(14) < 38.2 = trending). Only trade in trending markets to avoid whipsaw. Exit on opposite Donchian break or ATR stop (2.5x). Target 15-25 trades/year per symbol. Works in bull (breakouts) and bear (breakdowns) with regime filter reducing false signals in ranging markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')  # for Donchian trend filter
    df_1d = get_htf_data(prices, '1d')  # for chop regime (using 1d data)
    
    if len(df_1w) < 30 or len(df_1d) < 30:
        return np.zeros(n)
    
    # === 1w Donchian(20) for Breakout Signals ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Donchian upper/lower (20-period)
    donch_high = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    donch_high_aligned = align_htf_to_ltf(prices, df_1w, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1w, donch_low)
    
    # === 1d Indicators for Filters ===
    close = prices['close'].values
    volume = prices['volume'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Volume MA (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR (14-period) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Choppiness Index (14-period) for regime filter
    # CHOP = 100 * log10(sum(ATR) / (max(high) - min(low))) / log10(N)
    atr_14 = atr  # already calculated
    sum_atr = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop_raw = 100 * np.log10(sum_atr / (max_high - min_low + 1e-10)) / np.log10(14)
    # Handle division by zero or invalid values
    chop_raw = np.where((max_high - min_low) <= 0, 50.0, chop_raw)  # neutral when no range
    chop_raw = np.where(np.isnan(chop_raw), 50.0, chop_raw)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr[i]) or np.isnan(chop_raw[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ok = vol > 1.5 * vol_ma[i]  # volume confirmation
        chop_ok = chop_raw[i] < 38.2  # trending regime (CHOP < 38.2)
        
        if position == 0:
            # Long: break above 1w Donchian high with volume and trending regime
            if price > donch_high_aligned[i] and vol_ok and chop_ok:
                signals[i] = 0.25
                position = 1
            # Short: break below 1w Donchian low with volume and trending regime
            elif price < donch_low_aligned[i] and vol_ok and chop_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: ATR stoploss, opposite Donchian break, or chop regime change
            if (price < close[i-1] - 2.5 * atr[i] or 
                price < donch_low_aligned[i] or 
                chop_raw[i] >= 61.8):  # choppy regime -> exit
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: ATR stoploss, opposite Donchian break, or chop regime change
            if (price > close[i-1] + 2.5 * atr[i] or 
                price > donch_high_aligned[i] or 
                chop_raw[i] >= 61.8):  # choppy regime -> exit
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WeeklyDonchian_Breakout_VolumeRegime_V1"
timeframe = "1d"
leverage = 1.0