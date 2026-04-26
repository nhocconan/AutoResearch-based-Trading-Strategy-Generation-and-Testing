#!/usr/bin/env python3
"""
1d_Camarilla_Pivot_Volume_Regime_v1
Hypothesis: Daily Camarilla R3/S3 breakouts with weekly EMA50 trend filter, volume spike confirmation, and choppiness regime filter.
Daily timeframe reduces trade frequency to avoid fee drag while capturing sustained moves.
Camarilla levels derived from prior day's range provide high-probability S/R.
Volume spike confirms institutional participation. Choppiness filter avoids whipsaws in ranging markets.
Designed to work in both bull (trend continuation) and bear (mean reversion at extremes) markets.
Target: 30-80 total trades over 4 years (7-20/year).
"""

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
    
    # Load 1d data ONCE before loop for Camarilla calculation and trend context
    df_1d = get_htf_data(prices, '1d')
    
    # Load 1w data ONCE before loop for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Camarilla levels from prior 1d bar (HLC of previous day)
    # Camarilla: R3 = C + ((H-L)*1.1/4), S3 = C - ((H-L)*1.1/4)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla R3 and S3 for each 1d bar
    camarilla_r3 = close_1d + ((high_1d - low_1d) * 1.1 / 4)
    camarilla_s3 = close_1d - ((high_1d - low_1d) * 1.1 / 4)
    
    # Align Camarilla levels to 1d timeframe (already aligned, but for consistency)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume spike detection: volume > 2.0 * 50-period average volume
    avg_volume = pd.Series(volume).ewm(span=50, adjust=False, min_periods=50).mean().values
    volume_spike = volume > (2.0 * avg_volume)
    
    # Choppiness Index filter to avoid ranging markets
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate rolling sum of ATR(14) and range
    sum_atr_14 = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    max_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    range_14 = max_high_14 - min_low_14
    
    # Avoid division by zero
    chop = np.zeros_like(close)
    mask = (range_14 > 0) & (sum_atr_14 > 0)
    chop[mask] = 100 * np.log10(sum_atr_14[mask] / range_14[mask]) / np.log10(14)
    
    # Chop filter: only trade when market is trending (CHOP < 45)
    chop_filter = chop < 45
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need sufficient data for all indicators)
    start_idx = max(100, 50, 50, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(chop[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # 1w trend filter (EMA50)
        uptrend = close[i] > ema_50_1w_aligned[i]
        downtrend = close[i] < ema_50_1w_aligned[i]
        
        # Long logic: price breaks above camarilla R3 with volume spike + in uptrend + trending market
        if (close[i] > camarilla_r3_aligned[i] and 
            volume_spike[i] and 
            uptrend and 
            chop_filter[i]):
            if position != 1:
                signals[i] = 0.25
                position = 1
            else:
                signals[i] = 0.25
        # Short logic: price breaks below camarilla S3 with volume spike + in downtrend + trending market
        elif (close[i] < camarilla_s3_aligned[i] and 
              volume_spike[i] and 
              downtrend and 
              chop_filter[i]):
            if position != -1:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = -0.25
        # Exit conditions: price returns to opposite camarilla level or trend weakens or market becomes choppy
        elif position == 1 and (close[i] < camarilla_s3_aligned[i] or not uptrend or not chop_filter[i]):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (close[i] > camarilla_r3_aligned[i] or not downtrend or not chop_filter[i]):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Camarilla_Pivot_Volume_Regime_v1"
timeframe = "1d"
leverage = 1.0