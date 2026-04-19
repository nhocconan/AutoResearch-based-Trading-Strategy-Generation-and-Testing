#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_TRIX_VolumeSpike_ChopRegime_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for TRIX calculation (primary timeframe)
    df_4h = get_htf_data(prices, '4h')
    
    # Get 1d data for Chop filter (higher timeframe regime)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate TRIX(12) on 4h close
    close_4h = df_4h['close'].values
    # First EMA
    ema1 = pd.Series(close_4h).ewm(span=12, adjust=False, min_periods=12).mean().values
    # Second EMA
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean().values
    # Third EMA
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean().values
    # TRIX = 100 * (ema3 - ema3_prev) / ema3_prev
    trix_raw = 100 * (ema3 - np.roll(ema3, 1)) / np.roll(ema3, 1)
    trix_raw[0] = 0  # First value undefined
    
    # Calculate volume spike ratio (volume / volume_ma_20)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = np.where(volume_ma > 0, volume / volume_ma, 1.0)
    
    # Calculate Chop(14) on 1d high/low/close
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    # Avoid division by zero
    range_hl = highest_high - lowest_low
    range_hl = np.where(range_hl == 0, 1e-10, range_hl)
    chop = 100 * np.log10(pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values / range_hl) / np.log10(14)
    
    # Align all indicators to 4h timeframe
    trix_aligned = align_htf_to_ltf(prices, df_4h, trix_raw)
    volume_ratio_aligned = align_htf_to_ltf(prices, df_4h, volume_ratio)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop, additional_delay_bars=0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(trix_aligned[i]) or np.isnan(volume_ratio_aligned[i]) or 
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
            
        # Chop filter: trade only in ranging markets (Chop > 61.8 = choppy)
        chop_filter = chop_aligned[i] > 61.8
        
        if position == 0:
            # Long when TRIX turns positive with volume spike in choppy market
            if (trix_aligned[i] > 0 and trix_aligned[i-1] <= 0 and 
                volume_ratio_aligned[i] > 2.0 and chop_filter):
                signals[i] = 0.25
                position = 1
            # Short when TRIX turns negative with volume spike in choppy market
            elif (trix_aligned[i] < 0 and trix_aligned[i-1] >= 0 and 
                  volume_ratio_aligned[i] > 2.0 and chop_filter):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit when TRIX turns negative
            if trix_aligned[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit when TRIX turns positive
            if trix_aligned[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals