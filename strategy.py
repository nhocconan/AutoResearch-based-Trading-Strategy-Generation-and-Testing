#!/usr/bin/env python3
# 4h_donchian_volume_chop_v3
# Hypothesis: 4h Donchian(20) breakout with volume confirmation and chop regime filter.
# In trending markets (CHOP < 38.2), breakouts capture momentum; in ranging markets (CHOP > 61.8), 
# we avoid false breakouts. Volume > 1.5x 20-period average confirms institutional participation.
# Discrete sizing (0.0, ±0.30) limits fee churn. Target: 20-50 trades/year to avoid fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_volume_chop_v3"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for chop regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Chopiness Index (14-period) on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Absolute price change over 14 periods
    atr_1d = np.abs(close_1d[14:] - close_1d[:-14])
    atr_1d = np.concatenate([np.full(14, np.nan), atr_1d])
    
    # Chopiness Index: CHOP = 100 * log10(TR_sum / (ATR * n)) / log10(n)
    n_periods = 14
    chop_raw = 100 * np.log10(tr_14 / (atr_1d * n_periods)) / np.log10(n_periods)
    chop_1d = np.concatenate([np.full(14, np.nan), chop_raw])
    
    # Align chop to 4h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Donchian channels (20-period) on 4h
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        # Regime filter: only trade when not in extreme chop (CHOP between 38.2 and 61.8)
        # Avoid ranging markets where breakouts fail
        chop_value = chop_aligned[i]
        regime_filter = not (chop_value > 61.8 or chop_value < 38.2)
        
        if position == 1:  # Long position
            # Exit: price reaches Donchian low or volume dries up or regime changes to extreme chop
            if (close[i] <= donchian_low[i] or not volume_confirmed or 
                not regime_filter):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            # Exit: price reaches Donchian high or volume dries up or regime changes to extreme chop
            if (close[i] >= donchian_high[i] or not volume_confirmed or 
                not regime_filter):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.30
        else:  # Flat
            if volume_confirmed and regime_filter:
                # Long entry: price breaks above Donchian high with volume
                if close[i] > donchian_high[i]:
                    position = 1
                    signals[i] = 0.30
                # Short entry: price breaks below Donchian low with volume
                elif close[i] < donchian_low[i]:
                    position = -1
                    signals[i] = -0.30
    
    return signals