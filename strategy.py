#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + volume confirmation + 1d chop regime filter
# - Long when price breaks above Donchian(20) high with volume > 1.5x 20-bar avg AND 1d chop > 61.8 (range)
# - Short when price breaks below Donchian(20) low with volume > 1.5x 20-bar avg AND 1d chop > 61.8 (range)
# - Exit when price returns to Donchian(20) midpoint
# - Chop regime filter ensures trades occur in ranging markets where breakouts are more reliable
# - Volume confirmation reduces false breakouts
# - Targets ~30 trades/year (120 total over 4 years) to avoid fee drag
# - Works in both bull and bear markets by focusing on mean-reversion in ranges

name = "4h_1d_donchian_breakout_volume_chop_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d chop regime (Ehler's Choppiness Index)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index
    
    # ATR(14) sum
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Chop = 100 * log10(sum(atr14) / (hh14 - ll14)) / log10(14)
    chop = 100 * np.log10(atr_14 / (hh_14 - ll_14)) / np.log10(14)
    chop = np.where((hh_14 - ll_14) == 0, 100, chop)  # avoid division by zero
    chop = np.nan_to_num(chop, nan=100.0)
    
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Pre-compute 4h Donchian channels (20-period)
    high_4h = prices['high'].rolling(window=20, min_periods=20).max().values
    low_4h = prices['low'].rolling(window=20, min_periods=20).min().values
    donchian_mid = (high_4h + low_4h) / 2.0
    
    # Pre-compute 4h volume confirmation: > 1.5x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.5 * volume_20_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(high_4h[i]) or np.isnan(low_4h[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(volume_20_avg[i])):
            signals[i] = 0.0
            continue
        
        # Range regime: chop > 61.8 indicates ranging market
        in_range = chop_aligned[i] > 61.8
        
        if position == 0:  # Flat - look for new entries
            # Long signal: price breaks above Donchian high with volume spike in range
            if (prices['close'].iloc[i] > high_4h[i] and 
                vol_spike.iloc[i] and 
                in_range):
                position = 1
                signals[i] = 0.25
            # Short signal: price breaks below Donchian low with volume spike in range
            elif (prices['close'].iloc[i] < low_4h[i] and 
                  vol_spike.iloc[i] and 
                  in_range):
                position = -1
                signals[i] = -0.25
        else:  # Have position - look for exit
            # Exit when price returns to Donchian midpoint
            if position == 1 and prices['close'].iloc[i] < donchian_mid[i]:
                position = 0
                signals[i] = 0.0
            elif position == -1 and prices['close'].iloc[i] > donchian_mid[i]:
                position = 0
                signals[i] = 0.0
            # Hold position otherwise
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals