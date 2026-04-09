#!/usr/bin/env python3
# 4h_donchian_volume_chop_regime_v2
# Hypothesis: 4h Donchian(20) breakout with volume confirmation and chop regime filter.
# Long when price breaks above upper Donchian band with volume > 1.5x average in choppy market (chop > 61.8).
# Short when price breaks below lower Donchian band with volume > 1.5x average in choppy market (chop > 61.8).
# Exit when price reverses to opposite Donchian band or volume dries up.
# Uses discrete position sizing (±0.30) to minimize fee churn. Target: 20-50 trades/year.
# Works in both bull and bear markets by trading mean reversion in ranging conditions (chop > 61.8).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_volume_chop_regime_v2"
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
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate chop regime (14-period) from 1d data using correct formula
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range for ATR calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # First bar has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(14) - smoothed moving average
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Chop formula: CHOP = 100 * LOG10(ATR(14) / (LOG10(14) * (HH(14) - LL(14)))) / LOG10(14)
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero and log of zero
    atr_14_safe = np.where(atr_14 <= 0, 1e-10, atr_14)
    hl_range_14 = highest_high_14 - lowest_low_14
    hl_range_14_safe = np.where(hl_range_14 <= 0, 1e-10, hl_range_14)
    
    chop = 100 * np.log10(atr_14_safe) / (np.log10(14) * np.log10(hl_range_14_safe)) / np.log10(14)
    chop = np.where(np.isnan(chop) | np.isinf(chop), 50, chop)  # Default to neutral chop
    
    # Align chop to 4h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        # Chop regime: only trade when market is ranging (chop > 61.8 = trending threshold)
        chop_regime = chop_aligned[i] > 61.8
        
        if position == 1:  # Long position
            # Exit: price moves below Donchian low or volume dries up
            if close[i] < low_20[i] or not volume_confirmed:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            # Exit: price moves above Donchian high or volume dries up
            if close[i] > high_20[i] or not volume_confirmed:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.30
        else:  # Flat
            if volume_confirmed and chop_regime:
                # Long entry: price breaks above Donchian high with volume
                if close[i] > high_20[i]:
                    position = 1
                    signals[i] = 0.30
                # Short entry: price breaks below Donchian low with volume
                elif close[i] < low_20[i]:
                    position = -1
                    signals[i] = -0.30
    
    return signals