#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h TRIX momentum with volume spike and 1d chop regime filter.
# TRIX(9) > 0 and rising + volume spike (2x EMA60) + chop regime (range: CHOP > 61.8) = long
# TRIX(9) < 0 and falling + volume spike + chop regime (range: CHOP > 61.8) = short
# Chop regime filters out trending markets where TRIX whipsaws; range-bound markets favor mean reversion.
# Target: 80-120 total trades over 4 years (20-30/year) to balance opportunity and fee drag.

name = "4h_TRIX_VolumeSpike_ChopRange"
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
    
    # Get 1d data for chop regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # TRIX(9) on 4h close
    # TRIX = EMA(EMA(EMA(close, 9), 9), 9) - 1-period percent change
    ema1 = pd.Series(close).ewm(span=9, adjust=False, min_periods=9).mean().values
    ema2 = pd.Series(ema1).ewm(span=9, adjust=False, min_periods=9).mean().values
    ema3 = pd.Series(ema2).ewm(span=9, adjust=False, min_periods=9).mean().values
    trix = np.zeros_like(close)
    trix[1:] = (ema3[1:] - ema3[:-1]) / ema3[:-1] * 100
    
    # TRIX slope (rising/falling)
    trix_slope = np.zeros_like(trix)
    trix_slope[1:] = trix[1:] - trix[:-1]
    
    # Volume confirmation: 60-period volume spike (2.0x EMA)
    vol_ema = pd.Series(volume).ewm(span=60, adjust=False, min_periods=60).mean().values
    vol_confirm = volume > (vol_ema * 2.0)
    
    # Choppiness Index (CHOP) on 1d
    # CHOP = 100 * log10(sum(ATR(1)) / (max(high) - min(low))) / log10(n)
    atr_1d = np.zeros_like(close_1d)
    tr_1d = np.zeros_like(close_1d)
    tr_1d[0] = high_1d[0] - low_1d[0]
    for i in range(1, len(close_1d)):
        tr_1d[i] = max(high_1d[i] - low_1d[i], 
                       abs(high_1d[i] - close_1d[i-1]), 
                       abs(low_1d[i] - close_1d[i-1]))
    # ATR(1) is just TR
    atr_sum = np.zeros_like(close_1d)
    for i in range(14, len(close_1d)):
        atr_sum[i] = np.sum(tr_1d[i-13:i+1])
    max_high = np.zeros_like(close_1d)
    min_low = np.zeros_like(close_1d)
    for i in range(14, len(close_1d)):
        max_high[i] = np.max(high_1d[i-13:i+1])
        min_low[i] = np.min(low_1d[i-13:i+1])
    chop = np.full_like(close_1d, 50.0)  # default
    for i in range(14, len(close_1d)):
        if max_high[i] > min_low[i]:
            chop[i] = 100 * np.log10(atr_sum[i] / (max_high[i] - min_low[i])) / np.log10(14)
    
    # Chop regime: range when CHOP > 61.8
    chop_range = chop > 61.8
    
    # Align 1d indicators to 4h timeframe
    chop_range_aligned = align_htf_to_ltf(prices, df_1d, chop_range.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Ensure enough data for volume EMA and TRIX
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(trix[i]) or np.isnan(trix_slope[i]) or 
            np.isnan(vol_ema[i]) or np.isnan(chop_range_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: TRIX positive and rising + volume spike + chop range
            if (trix[i] > 0 and 
                trix_slope[i] > 0 and 
                vol_confirm[i] and 
                chop_range_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # Short entry: TRIX negative and falling + volume spike + chop range
            elif (trix[i] < 0 and 
                  trix_slope[i] < 0 and 
                  vol_confirm[i] and 
                  chop_range_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: TRIX turns negative or chop regime ends
            if (trix[i] <= 0 or 
                chop_range_aligned[i] <= 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: TRIX turns positive or chop regime ends
            if (trix[i] >= 0 or 
                chop_range_aligned[i] <= 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals