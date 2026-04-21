#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_Volume_ChopFilter_v1
Hypothesis: Price breaking above Camarilla R3 or below S3 from prior 1d session captures strong institutional breakouts with higher follow-through. Combined with volume spike (>2.0x 20-period MA) and choppiness regime filter (CHOP > 61.8 = range, avoid breakouts in chop). Designed for low trade frequency (~20-40/year) to minimize fee drag and work in both bull (breakout continuation) and bear (breakdown continuation) regimes by requiring strong momentum confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for Camarilla levels)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # === Camarilla levels from prior 1-day session (HLC of previous day) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla R3, S3 levels (stronger breakout signals)
    camarilla_r3 = close_1d + (high_1d - low_1d) * 1.1 / 4
    camarilla_s3 = close_1d - (high_1d - low_1d) * 1.1 / 4
    
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # === Volume spike filter (20-period) ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    # === Choppiness Index regime filter (14-period) ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Sum of True Range over 14 periods
    sum_tr = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: CHOP = 100 * log10(sum_tr / (hh - ll)) / log10(14)
    # Avoid division by zero when hh == ll
    hl_range = hh - ll
    chop = np.full(n, 50.0)  # default to neutral
    mask = (hl_range > 0) & (~np.isnan(sum_tr)) & (~np.isnan(hl_range))
    chop[mask] = 100 * np.log10(sum_tr[mask] / hl_range[mask]) / np.log10(14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(vol_ratio[i]) or np.isnan(chop[i]) or
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        price_high = prices['high'].iloc[i]
        price_low = prices['low'].iloc[i]
        vol_spike = vol_ratio[i]
        chop_val = chop[i]
        r3 = camarilla_r3_aligned[i]
        s3 = camarilla_s3_aligned[i]
        
        if position == 0:
            # Long: price breaks above R3 + volume spike > 2.0 + NOT in choppy regime (CHOP <= 61.8)
            if price_close > r3 and vol_spike > 2.0 and chop_val <= 61.8:
                signals[i] = 0.25
                position = 1
                entry_price = price_close
            # Short: price breaks below S3 + volume spike > 2.0 + NOT in choppy regime (CHOP <= 61.8)
            elif price_close < s3 and vol_spike > 2.0 and chop_val <= 61.8:
                signals[i] = -0.25
                position = -1
                entry_price = price_close
        
        elif position != 0:
            # Stoploss: 2.5 * ATR from entry (wider stop for less whipsaw)
            if position == 1:
                if price_close < entry_price - 2.5 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if price_close > entry_price + 2.5 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R3_S3_Breakout_Volume_ChopFilter_v1"
timeframe = "4h"
leverage = 1.0