#!/usr/bin/env python3
"""
4h Donchian Breakout + Volume Spike + Choppiness Regime Filter
Hypothesis: Donchian(20) breakouts capture strong momentum moves. Volume spike (>1.5x 20-period MA) confirms institutional participation. Choppiness Index > 61.8 ensures we only trade in ranging markets where breakouts are more likely to sustain (avoid false breakouts in strong trends). Works in bull markets (long breakouts) and bear markets (short breakouts) by being directionally agnostic. Discrete sizing (0.0, ±0.25) minimizes fee churn. Target: 20-50 trades/year on 4h.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for choppiness filter (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d Choppiness Index (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # align with index 0
    
    # ATR(14)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Highest high and lowest low over 14 periods
    hh_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Chop = 100 * log10(sum(atr14) / (hh14 - ll14)) / log10(14)
    sum_atr_14 = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    denominator = hh_14 - ll_14
    # Avoid division by zero
    chop_raw = np.where(denominator != 0, sum_atr_14 / denominator, 1.0)
    chop = 100 * np.log10(chop_raw) / np.log10(14)
    
    # Align chop to 4h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Calculate 20-period volume MA for volume spike confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Donchian(20) channels
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or
            np.isnan(vol_ma_20[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        vol_ma = vol_ma_20[i]
        chop_val = chop_aligned[i]
        upper_donch = donch_high[i]
        lower_donch = donch_low[i]
        
        # Volume spike: current volume > 1.5x 20-period MA
        volume_spike = curr_volume > (1.5 * vol_ma)
        
        # Choppiness regime: only trade when market is ranging (CHOP > 61.8)
        chop_regime = chop_val > 61.8
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above upper Donchian channel + volume spike + chop regime
            long_entry = (curr_close > upper_donch) and volume_spike and chop_regime
            # Short: price breaks below lower Donchian channel + volume spike + chop regime
            short_entry = (curr_close < lower_donch) and volume_spike and chop_regime
            
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
            # Exit: price falls below lower Donchian channel OR chop regime ends (trend starts)
            if (curr_close < lower_donch) or (chop_val <= 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price rises above upper Donchian channel OR chop regime ends (trend starts)
            if (curr_close > upper_donch) or (chop_val <= 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_VolumeSpike_ChopRegime"
timeframe = "4h"
leverage = 1.0