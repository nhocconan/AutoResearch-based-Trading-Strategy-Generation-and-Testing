#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_1dTrend_VolumeSpike_ChopFilter_v2
Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter, volume spike confirmation, and choppiness regime filter.
Only long when price breaks above upper Donchian(20) AND price > 1d EMA50 AND volume > 1.5 * 20-period EMA volume AND choppiness < 61.8 (trending regime).
Only short when price breaks below lower Donchian(20) AND price < 1d EMA50 AND volume spike AND choppiness < 61.8.
Uses discrete position sizing (0.0, ±0.25) to minimize fee churn. Designed for 75-200 total trades over 4 years (19-50/year).
Works in both bull and bear markets by combining price structure (Donchian) with trend (1d EMA) and regime filter (choppiness).
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
    
    # Donchian(20) on primary timeframe (4h)
    lookback = 20
    upper = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().shift(1).values
    lower = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().shift(1).values
    
    # 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: volume > 1.5 * 20-period EMA volume
    avg_volume = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (1.5 * avg_volume)
    
    # Choppiness regime filter (using 14-period)
    # CHOP = 100 * log10(sum(ATR,14) / (log10(HH-LL,14))) / log10(14)
    # CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    tr1 = np.maximum(high[1:], low[:-1])  # TR simplified for intraday
    tr1 = np.concatenate([[0], tr1])  # align lengths
    atr14 = pd.Series(tr1).rolling(window=14, min_periods=14).mean().values
    hh14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop_raw = 100 * np.log10(atr14 * 14 / (hh14 - ll14 + 1e-10)) / np.log10(14)
    chop_regime = chop_raw < 61.8  # trending regime
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = max(lookback, 50, 20, 14) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(volume_spike[i]) or np.isnan(chop_regime[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Discrete position sizing
        base_size = 0.25
        
        # Long logic: Donchian breakout up + 1d EMA50 up + volume spike + trending regime
        if close[i] > upper[i] and close[i] > ema_50_1d_aligned[i] and volume_spike[i] and chop_regime[i]:
            if position != 1:
                signals[i] = base_size
                position = 1
            else:
                signals[i] = base_size
        # Short logic: Donchian breakout down + 1d EMA50 down + volume spike + trending regime
        elif close[i] < lower[i] and close[i] < ema_50_1d_aligned[i] and volume_spike[i] and chop_regime[i]:
            if position != -1:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = -base_size
        # Exit conditions: Donchian breakout in opposite direction or loss of regime
        elif position == 1 and (close[i] < lower[i] or not chop_regime[i]):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (close[i] > upper[i] or not chop_regime[i]):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "4h_Donchian20_Breakout_1dTrend_VolumeSpike_ChopFilter_v2"
timeframe = "4h"
leverage = 1.0