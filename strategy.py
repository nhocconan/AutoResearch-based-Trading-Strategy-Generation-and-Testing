#!/usr/bin/env python3
"""
12h Donchian Breakout + Volume Confirmation + Regime Filter
Hypothesis: Donchian(20) breakouts on 12h timeframe with volume confirmation and 
choppiness regime filter will yield high-probability trades with low turnover.
Works in bull markets (breakouts up) and bear markets (breakouts down).
Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "14442_12h_donchian20_vol_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for Donchian and volume (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Load 1w data for choppiness regime filter
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # 12h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period) on 1d
    donch_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian to 12h
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma)
    
    # Choppiness index on 1w (14-period)
    # Chop = 100 * log10(sum(TR) / (HHV - LLV)) / log10(period)
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    hhvs = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    llvs = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / (hhvs - llvs + 1e-10)) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop)
    
    # Choppiness regime: Chop > 61.8 = ranging (mean revert), Chop < 38.2 = trending
    # For breakout strategy, we want trending markets (Chop < 38.2)
    chop_filter = chop_aligned < 38.2
    
    # ATR for stoploss (12h)
    tr1_12h = high - low
    tr2_12h = np.abs(high - np.roll(close, 1))
    tr3_12h = np.abs(low - np.roll(close, 1))
    tr_12h = np.maximum(tr1_12h, np.maximum(tr2_12h, tr3_12h))
    tr_12h[0] = tr1_12h[0]
    atr_12h = pd.Series(tr_12h).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(20, 14) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or
            np.isnan(vol_ma[i]) or np.isnan(atr_12h[i]) or np.isnan(chop_filter[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits
        if position == 1:  # long position
            # Exit: price breaks below Donchian low OR stoploss
            if (close[i] <= donch_low_aligned[i] or
                close[i] <= entry_price - 2.5 * atr_12h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above Donchian high OR stoploss
            if (close[i] >= donch_high_aligned[i] or
                close[i] >= entry_price + 2.5 * atr_12h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + volume + chop filter
            long_breakout = close[i] > donch_high_aligned[i]
            short_breakout = close[i] < donch_low_aligned[i]
            
            if long_breakout and vol_filter[i] and chop_filter[i]:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif short_breakout and vol_filter[i] and chop_filter[i]:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals