#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Choppiness_Index_MeanReversion_1dTrend"
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
    
    # 1d trend: EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Choppiness Index (14-period) for regime detection
    atr = np.zeros(n)
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate highest high and lowest low over 14 periods
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Chop = 100 * log10(sum(ATR14) / (max(HH14) - min(LL14))) / log10(14)
    atr_sum = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    range_14 = highest_high - lowest_low
    chop = 100 * np.log10(atr_sum / range_14) / np.log10(14)
    
    # Chop > 61.8 = ranging market (good for mean reversion)
    chopping = chop > 61.8
    
    # Bollinger Bands (20, 2) for mean reversion signals
    bb_mid = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    bb_std = pd.Series(close).rolling(window=20, min_periods=20).std().values
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(chopping[i]) or 
            np.isnan(bb_upper[i]) or np.isnan(bb_lower[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price at/below lower BB in ranging market + uptrend bias
            long_cond = (close[i] <= bb_lower[i]) and chopping[i] and (close[i] > ema_34_1d_aligned[i])
            # Short: price at/above upper BB in ranging market + downtrend bias
            short_cond = (close[i] >= bb_upper[i]) and chopping[i] and (close[i] < ema_34_1d_aligned[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price reaches middle BB or trend breaks
            if close[i] >= bb_mid[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reaches middle BB or trend breaks
            if close[i] <= bb_mid[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals