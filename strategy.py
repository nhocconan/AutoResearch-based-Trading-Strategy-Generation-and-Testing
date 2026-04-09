#!/usr/bin/env python3
# 4h_donchian_volume_chop_regime_v2
# Hypothesis: 4h Donchian breakout with volume confirmation and choppiness regime filter.
# Donchian(20) breakout captures trend continuation; volume >1.2x average confirms strength;
# Choppiness Index (14) > 61.8 filters ranging markets to avoid false breakouts.
# Discrete sizing (0.0, ±0.25) minimizes fee churn. Target: 20-40 trades/year.

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
    
    # 1d HTF data for choppiness regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range for 1d
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # ATR(14) and sum of TR for chop calculation
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    tr_sum_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Choppiness Index: CI = 100 * log10(sum(TR)/ATR) / log10(N)
    chop = 100 * np.log10(tr_sum_14 / atr_14) / np.log10(14)
    
    # Align 1d chop to 4h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Donchian channels (20-period) on 4h
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.2x 20-period average
        volume_confirmed = volume[i] > 1.2 * volume_ma[i]
        
        # Choppiness regime filter: only trade in trending markets (CHOP < 61.8)
        trending_market = chop_aligned[i] < 61.8
        
        if position == 1:  # Long position
            # Exit: price reaches Donchian lower band or volume dries up or chop too high
            if close[i] <= lowest_low[i] or not volume_confirmed or not trending_market:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price reaches Donchian upper band or volume dries up or chop too high
            if close[i] >= highest_high[i] or not volume_confirmed or not trending_market:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            if volume_confirmed and trending_market:
                # Long breakout: price breaks above Donchian upper band
                if close[i] > highest_high[i]:
                    position = 1
                    signals[i] = 0.25
                # Short breakdown: price breaks below Donchian lower band
                elif close[i] < lowest_low[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals