#!/usr/bin/env python3
# 4h_TRIX_VolumeSpike_ChopRegime
# Hypothesis: TRIX (triple smoothed EMA) captures momentum with reduced noise. Combined with volume spikes and Choppiness Index regime filter (CHOP > 61.8 = ranging, < 38.2 = trending), it avoids whipsaws in both bull and bear markets. Long when TRIX rising + volume spike + trending regime; short when TRIX falling + volume spike + trending regime. Uses 4h timeframe to limit trades and reduce fee drag.

name = "4h_TRIX_VolumeSpike_ChopRegime"
timeframe = "4h"
leverage = 1.0

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
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # TRIX: 15-period triple EMA of percent change
    # Step 1: EMA1 of close
    ema1 = pd.Series(close).ewm(span=15, adjust=False, min_periods=15).mean().values
    # Step 2: EMA2 of EMA1
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    # Step 3: EMA3 of EMA2
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    # TRIX = 100 * (EMA3_today - EMA3_yesterday) / EMA3_yesterday
    trix_raw = 100 * (ema3[1:] - ema3[:-1]) / ema3[:-1]
    trix = np.concatenate([[np.nan], trix_raw])
    # Smooth TRIX with 9-period EMA for signal line
    trix_smooth = pd.Series(trix).ewm(span=9, adjust=False, min_periods=9).mean().values
    
    # Choppiness Index (14-period) from daily data
    # CHOP = 100 * log10(sum(TR over n) / (max(HH) - min(LL))) / log10(n)
    # Using daily high/low/close
    dh = df_1d['high'].values
    dl = df_1d['low'].values
    dc = df_1d['close'].values
    # True Range
    tr1 = dh[1:] - dl[1:]
    tr2 = np.abs(dh[1:] - dc[:-1])
    tr3 = np.abs(dl[1:] - dc[:-1])
    tr_daily = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    # Sum of TR over 14 days
    tr_sum = pd.Series(tr_daily).rolling(window=14, min_periods=14).sum().values
    # Max high and min low over 14 days
    max_hh = pd.Series(dh).rolling(window=14, min_periods=14).max().values
    min_ll = pd.Series(dl).rolling(window=14, min_periods=14).min().values
    # Choppiness Index
    chop_raw = 100 * np.log10(tr_sum / (max_hh - min_ll)) / np.log10(14)
    chop = chop_raw  # already aligned to daily index
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Volume spike: volume > 2.0 x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(15+15+15+9, 20, 14)  # Ensure all indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(trix_smooth[i]) or np.isnan(chop_aligned[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: TRIX rising + volume spike + trending regime (CHOP < 38.2)
            if trix_smooth[i] > trix_smooth[i-1] and volume_spike[i] and chop_aligned[i] < 38.2:
                signals[i] = 0.25
                position = 1
            # Short: TRIX falling + volume spike + trending regime (CHOP < 38.2)
            elif trix_smooth[i] < trix_smooth[i-1] and volume_spike[i] and chop_aligned[i] < 38.2:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if TRIX turns down or regime becomes ranging (CHOP > 61.8)
            if trix_smooth[i] < trix_smooth[i-1] or chop_aligned[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if TRIX turns up or regime becomes ranging (CHOP > 61.8)
            if trix_smooth[i] > trix_smooth[i-1] or chop_aligned[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals