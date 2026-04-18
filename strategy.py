#!/usr/bin/env python3
"""
4h_PriceChannel_Breakout_Volume_Regime
Hypothesis: Breakout above/below 4h Donchian channel (20) with volume confirmation and 1d chop regime filter.
Long when price breaks above 4h DC upper with volume > 1.5x avg and 1d chop > 61.8 (range).
Short when price breaks below 4h DC lower with volume > 1.5x avg and 1d chop > 61.8.
Position size 0.25. Designed for mean-reversion in choppy markets (works in bull/bear).
Target: 20-40 trades/year per symbol to avoid fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4h Donchian channel (20-period)
    dc_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    dc_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 1d chop regime filter (choppiness index > 61.8 = range)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(np.roll(close_1d, 1) - low_1d)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = high_1d[0] - low_1d[0]
    
    # ATR(14) and sum of true ranges
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Chop = 100 * log10(sum(tr)/(atr*14)) / log10(14)
    chop = 100 * np.log10(tr_sum / (atr_14 * 14)) / np.log10(14)
    chop = np.where(tr_sum > 0, chop, 50)  # avoid division by zero
    
    chop_align = align_htf_to_ltf(prices, df_1d, chop)
    
    # Volume confirmation: current > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # need enough for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(dc_high[i]) or np.isnan(dc_low[i]) or 
            np.isnan(chop_align[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        in_range = chop_align[i] > 61.8  # chop > 61.8 = ranging market
        
        if position == 0:
            # Long: break above DC upper in ranging market with volume
            if close[i] > dc_high[i] and vol_confirm and in_range:
                signals[i] = 0.25
                position = 1
            # Short: break below DC lower in ranging market with volume
            elif close[i] < dc_low[i] and vol_confirm and in_range:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns below DC upper or chop drops (trending)
            if close[i] < dc_high[i] or chop_align[i] <= 61.8:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns above DC lower or chop drops (trending)
            if close[i] > dc_low[i] or chop_align[i] <= 61.8:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_PriceChannel_Breakout_Volume_Regime"
timeframe = "4h"
leverage = 1.0