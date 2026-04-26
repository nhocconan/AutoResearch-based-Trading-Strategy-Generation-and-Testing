#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_VolumeSpike_HTFTrend_RegimeFilter
Hypothesis: 4h Donchian(20) breakout with volume spike, HTF (1d) EMA50 trend filter, and choppiness regime filter.
Long when: price breaks above Donchian upper + 1d EMA50 uptrend + volume > 2.0 * avg volume + chop > 61.8 (range).
Short when: price breaks below Donchian lower + 1d EMA50 downtrend + volume > 2.0 * avg volume + chop > 61.8.
Exit: ATR trailing stop (2.5 * ATR) or Donchian opposite touch.
Uses discrete 0.30 position size. Targets 30-50 trades/year.
Works in bull (breakouts) and bear (range regime filters whipsaws).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 1d HTF for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_avg)
    
    # Choppiness Index (CHOP) regime filter - 14-period
    atr_period = 14
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    max_high = pd.Series(high).rolling(window=atr_period, min_periods=atr_period).max().values
    min_low = pd.Series(low).rolling(window=atr_period, min_periods=atr_period).min().values
    
    # Avoid division by zero
    chop_raw = 100 * np.log10(atr * np.sqrt(atr_period) / (max_high - min_low)) / np.log10(atr_period)
    chop = np.where((max_high - min_low) > 0, chop_raw, 50.0)
    chop_regime = chop > 61.8  # ranging regime
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    atr_stop = 0.0
    
    # Warmup: need 20 for Donchian/volume, 50 for 1d EMA, 14 for ATR/CHOP
    start_idx = max(20, 50, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_spike[i]) or
            np.isnan(chop_regime[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        size = 0.30  # Fixed position size
        
        if position == 0:
            # Flat - look for breakout with trend, volume, and regime confirmation
            long_entry = (close_val > donchian_upper[i]) and \
                       (ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1]) and \
                       volume_spike[i] and \
                       chop_regime[i]
            short_entry = (close_val < donchian_lower[i]) and \
                        (ema_50_1d_aligned[i] < ema_50_1d_aligned[i-1]) and \
                        volume_spike[i] and \
                        chop_regime[i]
            
            if long_entry:
                signals[i] = size
                position = 1
                entry_price = close_val
                atr_stop = entry_price - 2.5 * atr[i]
            elif short_entry:
                signals[i] = -size
                position = -1
                entry_price = close_val
                atr_stop = entry_price + 2.5 * atr[i]
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - update ATR stop and check exit conditions
            atr_stop = max(atr_stop, close_val - 2.5 * atr[i])
            if close_val <= atr_stop or close_val < donchian_lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short - update ATR stop and check exit conditions
            atr_stop = min(atr_stop, close_val + 2.5 * atr[i])
            if close_val >= atr_stop or close_val > donchian_upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Donchian20_Breakout_VolumeSpike_HTFTrend_RegimeFilter"
timeframe = "4h"
leverage = 1.0