#!/usr/bin/env python3
"""
4h_12h_TRIX_Volume_Regime_v1
Hypothesis: TRIX momentum on 12h timeframe combined with volume confirmation and Choppiness regime filter on 4h provides reliable trend signals.
TRIX filters noise, volume confirms institutional interest, and Choppiness identifies trending vs ranging markets.
Works in both bull and bear markets by adapting to regime. Target: 20-30 trades per year (80-120 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_TRIX_Volume_Regime_v1"
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
    
    # Get 12H data for TRIX
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    daily_close = df_12h['close'].values
    
    # === TRIX CALCULATION (12-period on 12h) ===
    # EMA1
    ema1 = pd.Series(daily_close).ewm(span=12, adjust=False, min_periods=12).mean().values
    # EMA2
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean().values
    # EMA3
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean().values
    # TRIX = (EMA3 - prev EMA3) / prev EMA3 * 100
    trix_raw = np.zeros_like(ema3)
    trix_raw[1:] = (ema3[1:] - ema3[:-1]) / (ema3[:-1] + 1e-10) * 100
    trix_signal = pd.Series(trix_raw).ewm(span=12, adjust=False, min_periods=12).mean().values
    trix_4h = align_htf_to_ltf(prices, df_12h, trix_signal)
    
    # === CHOPPINESS INDEX (14-period on 4h) ===
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    # Set first TR to high-low to avoid roll issue
    tr[0] = tr1[0]
    
    # ATR14
    atr = np.full(n, np.nan)
    if n >= 14:
        atr_sum = np.sum(tr[:14])
        atr[13] = atr_sum / 14
        for i in range(14, n):
            atr_sum = atr_sum - tr[i-14] + tr[i]
            atr[i] = atr_sum / 14
    
    # Sum of ATR over 14 periods
    atr_sum = np.full(n, np.nan)
    if n >= 14:
        atr_sum[13] = np.sum(atr[0:14])  # First valid sum
        for i in range(14, n):
            atr_sum[i] = atr_sum[i-1] - atr[i-14] + atr[i]
    
    # Max and min close over 14 periods
    max_close = np.full(n, np.nan)
    min_close = np.full(n, np.nan)
    if n >= 14:
        max_close[13] = np.max(close[0:14])
        min_close[13] = np.min(close[0:14])
        for i in range(14, n):
            max_close[i] = max(max_close[i-1], close[i])
            min_close[i] = min(min_close[i-1], close[i])
    
    # Chop = 100 * log10(sum(ATR14) / (max_close - min_close)) / log10(14)
    chop = np.full(n, np.nan)
    if n >= 14:
        denominator = max_close - min_close
        # Avoid division by zero
        denominator = np.where(denominator == 0, 1e-10, denominator)
        chop = 100 * np.log10(atr_sum / denominator) / np.log10(14)
    
    # === VOLUME SPIKE (2x 20-period average on 4h) ===
    vol_ma = np.full(n, np.nan)
    if n >= 20:
        vol_sum = np.sum(volume[:20])
        vol_ma[19] = vol_sum / 20
        for i in range(20, n):
            vol_sum = vol_sum - volume[i-20] + volume[i]
            vol_ma[i] = vol_sum / 20
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if any data invalid
        if (np.isnan(trix_4h[i]) or np.isnan(chop[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Regime filter: Chop < 38.2 = trending (use TRIX), Chop > 61.8 = ranging (avoid)
        trending_regime = chop[i] < 38.2
        
        # TRIX signals
        trix_bullish = trix_4h[i] > 0 and trix_4h[i] > trix_4h[i-1]  # Rising above zero
        trix_bearish = trix_4h[i] < 0 and trix_4h[i] < trix_4h[i-1]  # Falling below zero
        
        # Entry conditions with volume confirmation and regime filter
        long_entry = trending_regime and trix_bullish and vol_spike[i]
        short_entry = trending_regime and trix_bearish and vol_spike[i]
        
        # Exit conditions: opposite TRIX signal or chop exceeds threshold (range developing)
        long_exit = not trending_regime or (trix_4h[i] < 0)  # Exit long when TRIX turns negative or market ranges
        short_exit = not trending_regime or (trix_4h[i] > 0)  # Exit short when TRIX turns positive or market ranges
        
        # Signal logic
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals