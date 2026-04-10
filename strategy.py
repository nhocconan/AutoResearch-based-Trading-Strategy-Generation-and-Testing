#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume spike and choppiness regime filter
# - Long when price breaks above Donchian(20) high + 1d volume > 2.0x 20-bar avg + CHOP(14) > 61.8 (range)
# - Short when price breaks below Donchian(20) low + 1d volume > 2.0x 20-bar avg + CHOP(14) > 61.8 (range)
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Targets ~25 trades/year (100 total over 4 years) to avoid fee drag
# - CHOP filter ensures mean-reversion logic in ranging markets (2025 is bearish/ranging)
# - Volume confirmation ensures institutional participation
# - Donchian breakouts capture momentum after consolidation

name = "4h_1d_donchian_breakout_volume_chop_v1"
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
    
    # Pre-compute 1d HTF indicators
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1d EMA(50) for trend context (not used in entry but for regime)
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1d volume confirmation: > 2.0x 20-period average
    avg_volume_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = volume_1d > (2.0 * avg_volume_20_1d)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
    # 1d Choppiness Index: CHOP > 61.8 = ranging market (mean revert)
    # CHOP = 100 * log10(sum(ATR(14)) / log10(highest_high - lowest_low)) / log10(14)
    tr1 = np.maximum(high_1d[1:] - low_1d[:-1], np.abs(high_1d[1:] - close_1d[:-1]))
    tr2 = np.maximum(np.abs(low_1d[1:] - close_1d[:-1]), tr1)
    tr = np.concatenate([[np.nan], tr2])  # align with index
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    hh_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    range_14 = hh_14 - ll_14
    chop_raw = np.where(range_14 > 0, 
                        100 * np.log10(pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values) / 
                        np.log10(14) / np.log10(range_14), 50)
    chop_1d = chop_raw
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Pre-compute 4h Donchian channels
    highest_high_20 = prices['high'].rolling(window=20, min_periods=20).max().values
    lowest_low_20 = prices['low'].rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_spike_1d_aligned[i]) or 
            np.isnan(chop_1d_aligned[i]) or np.isnan(highest_high_20[i]) or 
            np.isnan(lowest_low_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long signal: Donchian breakout above + volume spike + choppy (range) market
            if (prices['close'].iloc[i] > highest_high_20[i] and 
                vol_spike_1d_aligned[i] and 
                chop_1d_aligned[i] > 61.8):
                position = 1
                signals[i] = 0.25
            # Short signal: Donchian breakdown below + volume spike + choppy (range) market
            elif (prices['close'].iloc[i] < lowest_low_20[i] and 
                  vol_spike_1d_aligned[i] and 
                  chop_1d_aligned[i] > 61.8):
                position = -1
                signals[i] = -0.25
        else:  # Have position - look for exit
            # Exit when price returns to midpoint of Donchian channel (mean reversion)
            midpoint = (highest_high_20[i] + lowest_low_20[i]) / 2.0
            if position == 1 and prices['close'].iloc[i] < midpoint:
                position = 0
                signals[i] = 0.0
            elif position == -1 and prices['close'].iloc[i] > midpoint:
                position = 0
                signals[i] = 0.0
            # Hold position otherwise
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals