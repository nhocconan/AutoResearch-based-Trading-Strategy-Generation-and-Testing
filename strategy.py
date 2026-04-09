#!/usr/bin/env python3
# 4h_donchian_breakout_volume_chop_v3
# Hypothesis: 4h Donchian channel breakout with volume confirmation and choppiness regime filter.
# Works in bull/bear: Donchian captures breakouts, volume confirms validity, chop filter avoids whipsaws in ranging markets.
# Target: 20-50 trades/year (80-200 total over 4 years).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_volume_chop_v3"
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
    
    # 1d HTF data for choppiness index
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range for 1d
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = tr2[0] = tr3[0] = 0  # First period has no prior close
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Choppiness Index (14-period)
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    hh = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / (hh - ll)) / np.log10(14)
    # Handle division by zero or invalid cases
    chop = np.where((hh - ll) == 0, 50, chop)  # Neutral when no range
    chop = np.where(np.isnan(chop), 50, chop)   # Neutral when NaN
    
    # Align choppiness to 4h timeframe (completed 1d bar only)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Donchian Channel (20-period) on 4h
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade when market is trending (chop < 61.8)
        is_trending = chop_aligned[i] < 61.8
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian low OR chop becomes too high (ranging)
            if close[i] < donchian_low[i] or chop_aligned[i] > 61.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian high OR chop becomes too high (ranging)
            if close[i] > donchian_high[i] or chop_aligned[i] > 61.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            if is_trending and not np.isnan(volume_ma[i]):
                volume_confirmed = volume[i] > 1.5 * volume_ma[i]
                
                if volume_confirmed:
                    # Long: price breaks above Donchian high
                    if close[i] > donchian_high[i]:
                        position = 1
                        signals[i] = 0.25
                    # Short: price breaks below Donchian low
                    elif close[i] < donchian_low[i]:
                        position = -1
                        signals[i] = -0.25
    
    return signals