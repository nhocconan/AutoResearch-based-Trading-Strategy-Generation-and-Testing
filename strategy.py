#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_TRIX_VolumeSpike_ChopFilter_v1"
timeframe = "4h"
load
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once before loop
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate TRIX on 1d close: TRIX = EMA(EMA(EMA(close,12),12),12) then % change
    close_series = pd.Series(close_1d)
    ema1 = close_series.ewm(span=12, adjust=False, min_periods=12).mean()
    ema2 = ema1.ewm(span=12, adjust=False, min_periods=12).mean()
    ema3 = ema2.ewm(span=12, adjust=False, min_periods=12).mean()
    trix_raw = 100 * (ema3 / ema3.shift(1) - 1)
    trix = trix_raw.fillna(0).values
    
    # Align TRIX to 4h timeframe
    trix_aligned = align_htf_to_ltf(prices, df_1d, trix)
    
    # Volume spike filter: current volume > 2x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Choppiness index on 4h: CHOP = 100 * log10(sum(ATR(14)) / (max(high)-min(low))) / log10(14)
    tr = np.maximum(high[1:], close[:-1]) - np.minimum(low[1:], close[:-1])
    tr = np.concatenate([[0], tr])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values / 
                          (highest_high - lowest_low)) / np.log10(14)
    chop = np.nan_to_num(chop, nan=50.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        if np.isnan(trix_aligned[i]) or np.isnan(vol_ma_20[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        chop_val = chop[i]
        
        # Volume filter
        volume_ok = vol > 2.0 * vol_ma
        
        # Chop filter: trending when CHOP < 38.2
        trending = chop_val < 38.2
        
        if position == 0:
            # Long: TRIX crosses above 0 with volume and trending market
            if i > 0 and trix_aligned[i-1] <= 0 and trix_aligned[i] > 0 and volume_ok and trending:
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses below 0 with volume and trending market
            elif i > 0 and trix_aligned[i-1] >= 0 and trix_aligned[i] < 0 and volume_ok and trending:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: TRIX crosses below 0 or market becomes choppy
            if trix_aligned[i] < 0 or chop_val >= 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: TRIX crosses above 0 or market becomes choppy
            if trix_aligned[i] > 0 or chop_val >= 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals