#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_VolumeSpike_1dATRRegime_v1
Hypothesis: Trade Donchian(20) breakouts on 4h with volume confirmation (2.0x median) and 1d ATR regime filter (low ATR = mean reversion bias, high ATR = trend follow). Only long when price > upper band and ATR(1d) rising, short when price < lower band and ATR(1d) falling. Uses ATR(4h) trailing stop (1.5x ATR). Designed to work in bull/bear by adapting to volatility regime. Target: 20-30 trades/year on 4h.
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
    
    # Get 1d data for ATR regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d ATR(14) for volatility regime
    tr_1d = np.maximum(df_1d['high'] - df_1d['low'], 
                       np.maximum(np.abs(df_1d['high'] - np.roll(df_1d['close'], 1)), 
                                  np.abs(df_1d['low'] - np.roll(df_1d['close'], 1))))
    tr_1d[0] = df_1d['high'].iloc[0] - df_1d['low'].iloc[0]
    atr_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # 1d ATR(50) for regime comparison
    atr_1d_50 = pd.Series(tr_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    atr_ratio_1d = atr_1d / np.where(atr_1d_50 > 0, atr_1d_50, np.nan)
    # Low volatility regime (mean reversion favor): ATR ratio < 0.8
    # High volatility regime (trend follow favor): ATR ratio > 1.2
    low_vol_regime = atr_ratio_1d < 0.8
    high_vol_regime = atr_ratio_1d > 1.2
    
    # Donchian(20) on 4h
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume confirmation: 2.0x median volume (20-period)
    vol_median = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    # ATR(14) for trailing stop (4h)
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align HTF indicators to 4h timeframe
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    atr_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio_1d)
    low_vol_regime_aligned = align_htf_to_ltf(prices, df_1d, low_vol_regime.astype(float))
    high_vol_regime_aligned = align_htf_to_ltf(prices, df_1d, high_vol_regime.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup: max of Donchian(20), volume median (20), ATR (14), ATR 1d (14,50)
    start_idx = max(lookback, 20, 14, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or
            np.isnan(vol_median[i]) or
            np.isnan(atr[i]) or
            np.isnan(atr_1d_aligned[i]) or
            np.isnan(atr_ratio_1d_aligned[i]) or
            np.isnan(low_vol_regime_aligned[i]) or
            np.isnan(high_vol_regime_aligned[i])):
            # Hold current position
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        highest_high_val = highest_high[i]
        lowest_low_val = lowest_low[i]
        close_val = close[i]
        volume_val = volume[i]
        vol_median_val = vol_median[i]
        atr_val = atr[i]
        atr_1d_val = atr_1d_aligned[i]
        atr_ratio_val = atr_ratio_1d_aligned[i]
        low_vol = low_vol_regime_aligned[i] > 0.5
        high_vol = high_vol_regime_aligned[i] > 0.5
        
        if position == 0:
            # Long: break above upper Donchian band with volume spike
            # In low vol regime: mean reversion bias (still take breakout but tighter stop)
            # In high vol regime: trend follow bias
            long_signal = (close_val > highest_high_val) and \
                          (volume_val > 2.0 * vol_median_val)
            
            # Short: break below lower Donchian band with volume spike
            short_signal = (close_val < lowest_low_val) and \
                           (volume_val > 2.0 * vol_median_val)
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
                highest_since_entry = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
                lowest_since_entry = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            highest_since_entry = max(highest_since_entry, high_val)
            # ATR trailing stop: 1.5x ATR in low vol, 2.0x ATR in high vol (wider stop in trending markets)
            atr_multiplier = 1.5 if low_vol else 2.0
            if close_val < highest_since_entry - atr_multiplier * atr_val:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            lowest_since_entry = min(lowest_since_entry, low_val)
            # ATR trailing stop: 1.5x ATR in low vol, 2.0x ATR in high vol
            atr_multiplier = 1.5 if low_vol else 2.0
            if close_val > lowest_since_entry + atr_multiplier * atr_val:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Donchian20_Breakout_VolumeSpike_1dATRRegime_v1"
timeframe = "4h"
leverage = 1.0