#!/usr/bin/env python3
"""
4h Donchian(20) Breakout + 1d ATR Regime Filter + Volume Spike
Hypothesis: Donchian channel breakouts capture strong momentum. 1d ATR regime filter avoids low-volatility chop. Volume spike confirms institutional participation. Works in bull/bear via discrete sizing (0.25) and ATR-based stops.
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
    
    # Load 1d data ONCE before loop for ATR regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d ATR(14) for regime filter
    def calculate_atr(high_arr, low_arr, close_arr, window=14):
        tr1 = np.abs(high_arr[1:] - low_arr[1:])
        tr2 = np.abs(high_arr[1:] - close_arr[:-1])
        tr3 = np.abs(low_arr[1:] - close_arr[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])
        atr = pd.Series(tr).ewm(span=window, adjust=False, min_periods=window).mean().values
        return atr
    
    atr_1d = calculate_atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # 1d ATR percentile rank (20-period) to detect high/low vol regimes
    atr_percentile = pd.Series(atr_1d).rolling(window=20, min_periods=20).rank(pct=True).values
    atr_percentile_aligned = align_htf_to_ltf(prices, df_1d, atr_percentile)
    
    # Donchian(20) from 4h data
    def calculate_donchian(high_arr, low_arr, window=20):
        upper = pd.Series(high_arr).rolling(window=window, min_periods=window).max().values
        lower = pd.Series(low_arr).rolling(window=window, min_periods=window).min().values
        return upper, lower
    
    donch_upper, donch_lower = calculate_donchian(high, low, window=20)
    
    # Volume confirmation: current volume > 1.8 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Donchian(20) and ATR warmup
    start_idx = max(30, 20)  # ATR needs ~14, Donchian needs 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]) or 
            np.isnan(atr_1d_aligned[i]) or np.isnan(atr_percentile_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        vol_spike = volume_spike[i]
        
        # Regime filter: only trade when volatility is elevated (ATR percentile > 0.4)
        high_vol_regime = atr_percentile_aligned[i] > 0.4
        
        if position == 0:
            # Look for entry signals - require: Donchian breakout + volume spike + high vol regime
            # Long: price breaks above Donchian upper AND volume spike AND high vol regime
            long_entry = (curr_high > donch_upper[i]) and vol_spike and high_vol_regime
            # Short: price breaks below Donchian lower AND volume spike AND high vol regime
            short_entry = (curr_low < donch_lower[i]) and vol_spike and high_vol_regime
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price falls below Donchian lower OR loss of high vol regime
            if (curr_low < donch_lower[i]) or (atr_percentile_aligned[i] <= 0.4):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price rises above Donchian upper OR loss of high vol regime
            if (curr_high > donch_upper[i]) or (atr_percentile_aligned[i] <= 0.4):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_1dATR_Regime_VolumeSpike"
timeframe = "4h"
leverage = 1.0