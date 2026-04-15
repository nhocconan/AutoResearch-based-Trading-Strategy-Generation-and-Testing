#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with volume confirmation and chop regime filter
# Long when price breaks above Camarilla R3 level + volume > 1.3x 20-period avg + chop > 61.8 (range)
# Short when price breaks below Camarilla S3 level + volume > 1.3x 20-period avg + chop > 61.8 (range)
# Uses 1d for Camarilla levels (more stable than intraday), 4h for execution timing
# Designed for mean reversion in ranging markets (chop > 61.8) with volume confirmation to avoid false breaks
# Target: 15-25 trades/year to minimize fee drag, works in both bull/bear via mean reversion logic

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) for filter
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d HTF data once before loop for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === 1d Indicator: Camarilla Pivot Levels (R3, S3) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Typical price for pivot calculation
    typical_price = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    camarilla_pivot = typical_price
    camarilla_r3 = camarilla_pivot + (range_1d * 1.1 / 4.0)
    camarilla_s3 = camarilla_pivot - (range_1d * 1.1 / 4.0)
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # === 4h Indicator: Choppiness Index (CHOP) for regime filter ===
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # True Range calculation
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period TR
    
    # ATR(14) - smoothed TR
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Sum of ATR over 14 periods
    sum_atr_14 = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    
    # Choppiness Index: CHOP = 100 * log10(sum(ATR)/range) / log10(14)
    max_high_14 = pd.Series(high_4h).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low_4h).rolling(window=14, min_periods=14).min().values
    range_14 = max_high_14 - min_low_14
    
    # Avoid division by zero
    chop = np.zeros_like(close_4h)
    mask = (range_14 > 0) & (sum_atr_14 > 0)
    chop[mask] = 100 * np.log10(sum_atr_14[mask] / range_14[mask]) / np.log10(14)
    
    # Align CHOP to 4h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_4h, chop)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.3x 20-period volume SMA
        vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.3)
        
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(chop_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade in ranging markets (CHOP > 61.8)
        if chop_aligned[i] <= 61.8:
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above Camarilla R3 level
        # 2. Volume confirmation
        if (close[i] > camarilla_r3_aligned[i]) and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below Camarilla S3 level
        # 2. Volume confirmation
        elif (close[i] < camarilla_s3_aligned[i]) and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "4h_Camarilla_R3S3_Volume_ChopFilter_v1"
timeframe = "4h"
leverage = 1.0