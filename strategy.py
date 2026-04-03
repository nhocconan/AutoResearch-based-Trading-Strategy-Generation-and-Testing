#!/usr/bin/env python3
"""
Experiment #2139: 6h Donchian(20) breakout + 12h ATR regime filter + volume confirmation
HYPOTHESIS: 6h Donchian breakouts capture swing momentum while ATR regime filter distinguishes trending (breakout) from ranging (fade) markets. Volume confirmation ensures institutional participation. Works in bull markets via trend-following breakouts and bear markets via mean-reversion fades at extremes when ATR low (ranging).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_2139_6h_donchian20_12h_atr_regime_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for ATR regime (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h ATR(14) for regime
    tr1_12h = high_12h - low_12h
    tr2_12h = np.abs(high_12h - np.roll(close_12h, 1))
    tr3_12h = np.abs(low_12h - np.roll(close_12h, 1))
    tr2_12h[0] = tr1_12h[0]
    tr3_12h[0] = tr1_12h[0]
    tr_12h = np.maximum(tr1_12h, np.maximum(tr2_12h, tr3_12h))
    atr_12h = pd.Series(tr_12h).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 12h ATR percentile rank (20-period lookback) for regime
    atr_percentile = np.full(n, np.nan)
    for i in range(20, len(atr_12h)):
        window = atr_12h[i-20:i+1]
        rank = np.sum(window <= atr_12h[i]) / len(window) * 100
        atr_percentile[i] = rank
    
    # Regime: 1 = trending (ATR > 70th percentile), 0 = ranging (ATR <= 70th)
    regime_12h = np.where(atr_percentile > 70, 1, 0)
    regime_12h_aligned = align_htf_to_ltf(prices, df_12h, regime_12h)
    
    # === 6h Indicators: Donchian(20), Volume MA(20) ===
    # Donchian channels
    high_ma = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_ma = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_upper = high_ma
    donchian_lower = low_ma
    
    # Volume MA for spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = 50  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(regime_12h_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit conditions:
                # 1. Trending regime: trail with 2*ATR stop
                # 2. Ranging regime: mean reversion at Donchian bands
                if regime_12h_aligned[i] == 1:  # Trending
                    if price < highest_since_entry - 2.0 * atr_12h[i]:
                        in_position = False
                        position_side = 0
                        signals[i] = 0.0
                    else:
                        signals[i] = SIZE
                else:  # Ranging
                    if price <= donchian_lower[i] or price >= donchian_upper[i]:
                        in_position = False
                        position_side = 0
                        signals[i] = 0.0
                    else:
                        signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit conditions:
                # 1. Trending regime: trail with 2*ATR stop
                # 2. Ranging regime: mean reversion at Donchian bands
                if regime_12h_aligned[i] == 1:  # Trending
                    if price > lowest_since_entry + 2.0 * atr_12h[i]:
                        in_position = False
                        position_side = 0
                        signals[i] = 0.0
                    else:
                        signals[i] = -SIZE
                else:  # Ranging
                    if price <= donchian_lower[i] or price >= donchian_upper[i]:
                        in_position = False
                        position_side = 0
                        signals[i] = 0.0
                    else:
                        signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Volume confirmation: require volume spike (> 1.5x average)
        volume_spike = vol_ratio[i] > 1.5
        
        if volume_spike:
            if regime_12h_aligned[i] == 1:  # Trending regime: breakout continuation
                # Long: price breaks above upper Donchian
                if price > donchian_upper[i]:
                    in_position = True
                    position_side = 1
                    entry_price = close[i]
                    highest_since_entry = high[i]
                    lowest_since_entry = low[i]
                    signals[i] = SIZE
                # Short: price breaks below lower Donchian
                elif price < donchian_lower[i]:
                    in_position = True
                    position_side = -1
                    entry_price = close[i]
                    highest_since_entry = high[i]
                    lowest_since_entry = low[i]
                    signals[i] = -SIZE
                else:
                    signals[i] = 0.0
            else:  # Ranging regime: fade at extremes
                # Long: price touches lower Donchian (mean reversion long)
                if price <= donchian_lower[i]:
                    in_position = True
                    position_side = 1
                    entry_price = close[i]
                    highest_since_entry = high[i]
                    lowest_since_entry = low[i]
                    signals[i] = SIZE
                # Short: price touches upper Donchian (mean reversion short)
                elif price >= donchian_upper[i]:
                    in_position = True
                    position_side = -1
                    entry_price = close[i]
                    highest_since_entry = high[i]
                    lowest_since_entry = low[i]
                    signals[i] = -SIZE
                else:
                    signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals