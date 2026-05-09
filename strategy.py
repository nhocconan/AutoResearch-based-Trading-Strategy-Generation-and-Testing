#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Choppiness_MeanReversion_1dATR_Stop"
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
    
    # Get daily data for ATR and Bollinger Bands
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Daily ATR(14) for stop loss and position sizing
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
    tr2 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, tr2)
    atr_1d = np.zeros_like(close_1d)
    atr_1d[14] = np.mean(tr[:14])
    for i in range(15, len(tr)+1):
        atr_1d[i-1] = (atr_1d[i-2] * 13 + tr[i-1]) / 14
    atr_1d = np.concatenate([np.full(14, np.nan), atr_1d])
    
    # Daily Bollinger Bands(20,2) for mean reversion signals
    close_series = pd.Series(close_1d)
    bb_mid = close_series.rolling(window=20, min_periods=20).mean().values
    bb_std = close_series.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    
    # Daily Choppiness Index(14) for regime filter
    atr_sum = np.zeros_like(close_1d)
    for i in range(14, len(tr)):
        atr_sum[i] = np.sum(tr[i-13:i+1])
    highest_high = np.maximum.accumulate(high_1d)
    lowest_low = np.minimum.accumulate(low_1d)
    hh_ll = highest_high - lowest_low
    chop = np.zeros_like(close_1d)
    chop[14:] = 100 * np.log10(atr_sum[14:] / hh_ll[14:]) / np.log10(14)
    
    # Align daily indicators to 4h timeframe
    bb_mid_aligned = align_htf_to_ltf(prices, df_1d, bb_mid)
    bb_upper_aligned = align_htf_to_ltf(prices, df_1d, bb_upper)
    bb_lower_aligned = align_htf_to_ltf(prices, df_1d, bb_lower)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # 4h Bollinger Bands for entry timing (optional refinement)
    close_4h_series = pd.Series(close)
    bb_mid_4h = close_4h_series.rolling(window=20, min_periods=20).mean().values
    bb_std_4h = close_4h_series.rolling(window=20, min_periods=20).std().values
    bb_upper_4h = bb_mid_4h + 2 * bb_std_4h
    bb_lower_4h = bb_mid_4h - 2 * bb_std_4h
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    atr_multiplier = 2.0  # ATR multiplier for stop loss
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(bb_mid_aligned[i]) or np.isnan(bb_upper_aligned[i]) or 
            np.isnan(bb_lower_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(atr_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Chop regime: mean reversion when chop > 61.8 (ranging market)
        chop_ok = chop_aligned[i] > 61.8
        
        if position == 0:
            # Long: Price touches lower Bollinger Band in ranging market
            if close[i] <= bb_lower_aligned[i] and chop_ok:
                signals[i] = 0.25
                position = 1
            # Short: Price touches upper Bollinger Band in ranging market
            elif close[i] >= bb_upper_aligned[i] and chop_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price reaches middle band or stop loss hit
            if close[i] >= bb_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price reaches middle band or stop loss hit
            if close[i] <= bb_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals