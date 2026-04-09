#!/usr/bin/env python3
# 4h_donchian_breakout_volume_chop_regime_v2
# Hypothesis: 4h Donchian(20) breakout with volume confirmation and chop regime filter.
# In trending markets (chop < 38.2), breakouts capture momentum. In ranging markets (chop > 61.8),
# fade the breakout as mean reversion. Volume spike confirms institutional participation.
# Discrete sizing (0.0, ±0.25) minimizes fee churn. Target: 100-200 total trades over 4 years.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_volume_chop_regime_v2"
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
    
    # 1d HTF data for chop regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Chop index (14-period) on 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # True range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close, 1))
    tr3 = np.abs(low_1d - np.roll(close, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # first bar
    tr2[0] = high_1d[0] - close[0]   # first bar
    tr3[0] = low_1d[0] - close[0]    # first bar
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Chop formula: 100 * log10(sum(TR14) / log10(14) * (max(H14) - min(L14))) / log10(14)
    chop_denom = np.log10(atr_14) * np.log10(14)
    chop_denom = np.where(chop_denom == 0, 1e-10, chop_denom)
    chop = 100 * np.log10((high_14 - low_14) / chop_denom) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Donchian channels (20-period) on 4h data
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.8x 20-period average
        volume_confirmed = volume[i] > 1.8 * volume_ma[i]
        
        if position == 1:  # Long position
            # Exit: price moves below Donchian low or volume dries up
            if close[i] < low_20[i] or not volume_confirmed:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price moves above Donchian high or volume dries up
            if close[i] > high_20[i] or not volume_confirmed:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            if volume_confirmed:
                # Regime-based logic
                chop_val = chop_aligned[i]
                
                if chop_val < 38.2:  # Strong trend - follow breakout
                    # Long breakout
                    if close[i] > high_20[i]:
                        position = 1
                        signals[i] = 0.25
                    # Short breakout
                    elif close[i] < low_20[i]:
                        position = -1
                        signals[i] = -0.25
                elif chop_val > 61.8:  # Strong range - fade breakout
                    # Fade upward breakout (sell)
                    if close[i] > high_20[i]:
                        position = -1
                        signals[i] = -0.25
                    # Fade downward breakout (buy)
                    elif close[i] < low_20[i]:
                        position = 1
                        signals[i] = 0.25
                # In transition zone (38.2 <= chop <= 61.8), stay flat
    
    return signals