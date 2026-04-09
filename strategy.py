#!/usr/bin/env python3
# 4h_donchian_volume_chop_regime_v1
# Hypothesis: 4h Donchian breakout with volume confirmation and choppiness regime filter.
# Long when price breaks above Donchian(20) high + volume > 1.5x average + choppy market (CHOP > 61.8).
# Short when price breaks below Donchian(20) low + volume > 1.5x average + choppy market (CHOP > 61.8).
# Uses 1d HTF for choppiness filter to avoid trending markets where breakouts fail.
# Discrete sizing (0.0, ±0.25) to minimize fee churn. Target: 20-50 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_volume_chop_regime_v1"
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
    
    # Donchian channels (20-period)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # 1d HTF data for choppiness regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range for choppiness calculation
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Sum of true ranges over 14 periods
    sum_tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Choppiness Index: CHOP = 100 * log10(sumTR14 / (ATR14 * 14)) / log10(14)
    chop_raw = 100 * np.log10(sum_tr_14 / (atr_1d * 14)) / np.log10(14)
    chop_1d = chop_raw  # Already calculated with proper min_periods via rolling
    
    # Align 1d choppiness to 4h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(volume_ma[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        # Regime filter: choppy market (CHOP > 61.8) - good for mean reversion/breakouts in range
        choppy_market = chop_aligned[i] > 61.8
        
        if position == 1:  # Long position
            # Exit: price reaches Donchian low or volume dries up or market trends
            if close[i] <= lowest_low[i] or not volume_confirmed or not choppy_market:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price reaches Donchian high or volume dries up or market trends
            if close[i] >= highest_high[i] or not volume_confirmed or not choppy_market:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            if volume_confirmed and choppy_market:
                # Long breakout: price breaks above Donchian high with volume
                if close[i] > highest_high[i]:
                    position = 1
                    signals[i] = 0.25
                # Short breakdown: price breaks below Donchian low with volume
                elif close[i] < lowest_low[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals