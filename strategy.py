#!/usr/bin/env python3
"""
4h Donchian(20) Breakout + 1d ATR Regime + Volume Spike
Hypothesis: Donchian channel breakouts capture strong momentum. ATR regime filter (ATR(7)/ATR(30) > 1.5) ensures we only trade in volatile regimes where breakouts are meaningful. Volume confirmation ensures participation. Works in bull (long on upper break) and bear (short on lower break). Target: 30-60 trades/year on 4h.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ATR regime and Donchian channels (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ATR(7) and ATR(30) for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index 0
    
    # ATR(7) and ATR(30)
    atr_7 = pd.Series(tr).rolling(window=7, min_periods=7).mean().values
    atr_30 = pd.Series(tr).rolling(window=30, min_periods=30).mean().values
    atr_ratio = atr_7 / atr_30  # ATR(7)/ATR(30) ratio
    
    # Align ATR ratio to 4h timeframe
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # Calculate 1d Donchian channels (20-period)
    donch_h = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donch_l = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donch_h_aligned = align_htf_to_ltf(prices, df_1d, donch_h)
    donch_l_aligned = align_htf_to_ltf(prices, df_1d, donch_l)
    
    # Calculate 4h volume MA for volume confirmation (20-period)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for ATR, Donchian, volume MA
    start_idx = max(30, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(atr_ratio_aligned[i]) or np.isnan(donch_h_aligned[i]) or 
            np.isnan(donch_l_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        atr_ratio_val = atr_ratio_aligned[i]
        donch_h_val = donch_h_aligned[i]
        donch_l_val = donch_l_aligned[i]
        vol_ma = vol_ma_20[i]
        
        # Volume confirmation: current volume > 2.0 * 20-period average
        volume_confirm = curr_volume > 2.0 * vol_ma
        
        # Regime filter: only trade when ATR(7)/ATR(30) > 1.5 (volatile regime)
        volatile_regime = atr_ratio_val > 1.5
        
        if position == 0:
            # Look for entry signals
            # Long: price > upper Donchian, volatile regime, volume confirmation
            long_entry = (curr_close > donch_h_val) and volatile_regime and volume_confirm
            # Short: price < lower Donchian, volatile regime, volume confirmation
            short_entry = (curr_close < donch_l_val) and volatile_regime and volume_confirm
            
            if long_entry:
                signals[i] = 0.30
                position = 1
            elif short_entry:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price crosses below lower Donchian (stop and reverse)
            if curr_close < donch_l_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Short position management
            # Exit: price crosses above upper Donchian (stop and reverse)
            if curr_close > donch_h_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

name = "4h_Donchian20_Breakout_1dATRRegime_VolumeSpike"
timeframe = "4h"
leverage = 1.0