#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_with_Regime_Filter
Hypothesis: Camarilla R1/S1 breakout on 4h with 1d volume spike and choppiness regime filter. 
Only trade breakouts in non-choppy markets (CHOP > 61.8 = ranging, CHOP < 38.2 = trending).
Uses tighter entry conditions to reduce trade count and avoid fee drag. Works in bull (trend breakouts) 
and bear (mean reversion in ranging markets) by adapting to regime.
"""

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
    
    # Calculate 1d Camarilla pivot levels (R1, S1)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    PP = (high_1d + low_1d + close_1d) / 3.0
    R1 = PP + (high_1d - low_1d) * 1.0 / 4.0
    S1 = PP - (high_1d - low_1d) * 1.0 / 4.0
    
    # Align Camarilla levels to 4h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_avg)
    
    # Choppiness Index: CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    # Using 14-period CHOP on 4h data
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar
    
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    hh14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    chop = np.zeros(n)
    for i in range(n):
        if i >= 13 and atr14[i] > 0 and hh14[i] != ll14[i]:
            chop[i] = 100 * np.log10(atr14[i] / (hh14[i] - ll14[i])) / np.log10(14)
        else:
            chop[i] = 50.0  # neutral
    
    # Regime filter: only trade when NOT choppy (CHOP < 61.8 for trending, or use mean reversion in chop)
    # For breakout strategy, we want trending markets: CHOP < 61.8
    not_choppy = chop < 61.8
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: need enough for all indicators
    start_idx = max(100, 20, 14)  # volume avg, chop
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or
            np.isnan(volume_spike[i]) or np.isnan(not_choppy[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        size = 0.25  # 25% position size
        
        if position == 0:
            # Flat - look for breakout with volume confirmation and regime filter
            # Only trade breakouts in non-choppy (trending) markets
            long_entry = (close_val > R1_aligned[i]) and volume_spike[i] and not_choppy[i]
            short_entry = (close_val < S1_aligned[i]) and volume_spike[i] and not_choppy[i]
            
            if long_entry:
                signals[i] = size
                position = 1
                entry_price = close_val
            elif short_entry:
                signals[i] = -size
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit on S1 retracement or if market becomes too choppy
            if close_val < S1_aligned[i] or chop[i] >= 61.8:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit on R1 retracement or if market becomes too choppy
            if close_val > R1_aligned[i] or chop[i] >= 61.8:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_with_Regime_Filter"
timeframe = "4h"
leverage = 1.0