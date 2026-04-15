#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with volume confirmation and chop regime filter
# Long when price breaks above 1d Camarilla R3 + volume > 1.3x 20-period avg + CHOP > 61.8 (range)
# Short when price breaks below 1d Camarilla S3 + volume > 1.3x 20-period avg + CHOP > 61.8 (range)
# Exit when price returns to 1d Camarilla pivot (mean reversion in ranging markets)
# Designed for low trade frequency (12-37/year) to minimize fee drag in bear markets (2025+ test)
# Uses 12h for signal generation, 1d for pivot levels and regime filter

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
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === 1d Indicator: Camarilla Pivot Levels ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Typical price for pivot calculation
    typical_price_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    pivot_1d = typical_price_1d
    r3_1d = pivot_1d + (range_1d * 1.1 / 2)
    s3_1d = pivot_1d - (range_1d * 1.1 / 2)
    
    # Align to 12h timeframe
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    # === 12h Indicator: Choppiness Index (CHOP) for regime filter ===
    # CHOP > 61.8 = ranging market (good for mean reversion)
    # CHOP < 38.2 = trending market
    high_12h = high
    low_12h = low
    close_12h = close
    
    # True Range
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # ATR(14)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Sum of TRUE Range over 14 periods
    sum_tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Choppiness Index: CHOP = 100 * log10(sum_tr_14 / (atr_14 * 14)) / log10(14)
    chop = 100 * np.log10(sum_tr_14 / (atr_14 * 14)) / np.log10(14)
    # Handle division by zero or invalid values
    chop = np.where((atr_14 * 14) > 0, chop, 50.0)  # Default to neutral when ATR is zero
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 50
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.3x 20-period volume SMA
        vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.3)
        
        # Skip if any required data is NaN
        if (np.isnan(pivot_1d_aligned[i]) or np.isnan(r3_1d_aligned[i]) or 
            np.isnan(s3_1d_aligned[i]) or np.isnan(chop[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above 1d Camarilla R3
        # 2. Volume confirmation
        # 3. Chop > 61.8 (ranging market - good for mean reversion plays)
        if (close[i] > r3_1d_aligned[i]) and vol_confirm and (chop[i] > 61.8):
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below 1d Camarilla S3
        # 2. Volume confirmation
        # 3. Chop > 61.8 (ranging market - good for mean reversion plays)
        elif (close[i] < s3_1d_aligned[i]) and vol_confirm and (chop[i] > 61.8):
            signals[i] = -0.25
        
        # === EXIT CONDITIONS ===
        # Exit when price returns to Camarilla pivot (mean reversion target)
        elif (np.sign(signals[i-1]) == 1 and close[i] <= pivot_1d_aligned[i]) or \
             (np.sign(signals[i-1]) == -1 and close[i] >= pivot_1d_aligned[i]):
            signals[i] = 0.0
        
        else:
            # Hold previous position
            signals[i] = signals[i-1]
    
    return signals

name = "12h_Camarilla_R3S3_Volume_Chop_Filter_v1"
timeframe = "12h"
leverage = 1.0