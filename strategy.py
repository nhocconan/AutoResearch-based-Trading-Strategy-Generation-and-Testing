#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R1/S1 breakout with volume confirmation and 1d choppiness regime filter.
Long when price breaks above Camarilla R1 AND volume > 1.3x average AND chop < 61.8 (trending).
Short when price breaks below Camarilla S1 AND volume > 1.3x average AND chop < 61.8.
Exit when price reverts to Camarilla midpoint (close) OR chop > 61.8 (choppy market).
Uses 4h for Camarilla calculation and 1d for choppiness filter to reduce whipsaw.
Target: 75-200 total trades over 4 years (19-50/year). Camarilla levels provide precise intraday
support/resistance, volume confirmation filters breakout validity, chop filter avoids ranging markets.
Works in bull markets (captures uptrends from R1) and bear markets (captures downtrends from S1).
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
    volume = prices['volume'].values
    
    # Get 4h data for Camarilla calculation
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Camarilla levels on 4h timeframe (based on previous day)
    # Camarilla: R1 = close + (high - low) * 1.1/12, S1 = close - (high - low) * 1.1/12
    # We need previous 4h bar's high/low/close to calculate today's levels
    prev_high_4h = np.roll(high_4h, 1)
    prev_low_4h = np.roll(low_4h, 1)
    prev_close_4h = np.roll(close_4h, 1)
    prev_high_4h[0] = np.nan
    prev_low_4h[0] = np.nan
    prev_close_4h[0] = np.nan
    
    rng = prev_high_4h - prev_low_4h
    camarilla_r1 = prev_close_4h + rng * 1.1 / 12
    camarilla_s1 = prev_close_4h - rng * 1.1 / 12
    camarilla_mid = prev_close_4h  # midpoint is previous close
    
    # Get 1d data for choppiness filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate choppiness index on 1d timeframe (14-period)
    high_1d_series = pd.Series(high_1d)
    low_1d_series = pd.Series(low_1d)
    close_1d_series = pd.Series(close_1d)
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    
    # ATR (14-period)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Highest high and lowest low over 14 periods
    hh = high_1d_series.rolling(window=14, min_periods=14).max().values
    ll = low_1d_series.rolling(window=14, min_periods=14).min().values
    
    # Chop = 100 * log10(sum(atr)/log(hh/ll)) / log10(14)
    sum_atr = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    ratio = hh / ll
    ratio = np.where(ratio <= 1, 1.001, ratio)  # avoid division by zero or log<=0
    chop = 100 * (np.log10(sum_atr) - np.log10(ratio)) / np.log10(14)
    
    # Align 4h Camarilla to 4h timeframe (no alignment needed for same TF)
    camarilla_r1_aligned = camarilla_r1
    camarilla_s1_aligned = camarilla_s1
    camarilla_mid_aligned = camarilla_mid
    
    # Align 1d chop to 4h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Volume average (20-period) on 4h
    volume_4h = df_4h['volume'].values
    volume_ma = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    volume_ma_aligned = align_htf_to_ltf(prices, df_4h, volume_ma)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(camarilla_mid_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(volume_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        r1 = camarilla_r1_aligned[i]
        s1 = camarilla_s1_aligned[i]
        mid = camarilla_mid_aligned[i]
        chop_val = chop_aligned[i]
        vol_ma = volume_ma_aligned[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Long: price > Camarilla R1 AND volume > 1.3x avg AND chop < 61.8 (trending)
            if price > r1 and vol > 1.3 * vol_ma and chop_val < 61.8:
                signals[i] = 0.25
                position = 1
            # Short: price < Camarilla S1 AND volume > 1.3x avg AND chop < 61.8 (trending)
            elif price < s1 and vol > 1.3 * vol_ma and chop_val < 61.8:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price < Camarilla midpoint OR chop > 61.8 (choppy market)
            if price < mid or chop_val > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price > Camarilla midpoint OR chop > 61.8 (choppy market)
            if price > mid or chop_val > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1S1_Volume_Chop_Filter"
timeframe = "4h"
leverage = 1.0