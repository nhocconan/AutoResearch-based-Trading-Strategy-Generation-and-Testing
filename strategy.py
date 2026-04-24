#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d volume spike and choppiness regime filter.
- Primary timeframe: 12h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for Donchian channels, volume average, and choppiness index.
- Donchian breakout: Long when price > 20-period high, Short when price < 20-period low.
- Volume confirmation: Current 12h volume > 2.0 * 20-period 1d volume average.
- Regime filter: Only trade when choppiness index (14) < 50 (trending market).
- Signal size: 0.25 discrete to minimize fee drag.
- Works in both bull and bear markets by trading breakouts in trending regimes only.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need sufficient data for calculations
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d Donchian channels (20-period)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:  # Need sufficient data for Donchian
        return np.zeros(n)
    
    # 20-period high and low for Donchian channels
    high_20 = pd.Series(df_1d['high'].values).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(df_1d['low'].values).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian channels to 12h timeframe
    high_20_aligned = align_htf_to_ltf(prices, df_1d, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_1d, low_20)
    
    # Calculate 1d volume average for confirmation (20-period)
    if len(df_1d) < 20:
        return np.zeros(n)
    
    vol_ma_20_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Calculate 1d choppiness index (14-period)
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # True Range calculation
    tr1 = pd.Series(df_1d['high'].values - df_1d['low'].values)
    tr2 = pd.Series(abs(df_1d['high'].values - df_1d['close'].values.shift(1)))
    tr3 = pd.Series(abs(df_1d['low'].values - df_1d['close'].values.shift(1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # ATR14
    atr14 = tr.rolling(window=14, min_periods=14).sum().values / 14  # Using sum/period for simplicity
    
    # Sum of absolute price changes over 14 periods
    price_change = pd.Series(abs(df_1d['close'].values - df_1d['close'].values.shift(1))).rolling(window=14, min_periods=14).sum().values
    
    # Choppiness Index: 100 * log10(atr14_sum / (price_change_sum)) / log10(14)
    # Avoid division by zero
    chop = np.full_like(price_change, 50.0)  # Default to neutral
    mask = (price_change > 0) & (atr14 > 0)
    chop[mask] = 100 * np.log10(atr14[mask] / price_change[mask]) / np.log10(14)
    
    # Align choppiness index to 12h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 14)  # Need 20 for Donchian/volume, 14 for chop
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(high_20_aligned[i]) or np.isnan(low_20_aligned[i]) or
            np.isnan(vol_ma_20_1d_aligned[i]) or np.isnan(chop_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Regime filter: only trade in trending market (chop < 50)
        trending_regime = chop_aligned[i] < 50
        
        # Volume confirmation: current volume > 2.0 * 20-period average volume
        volume_confirm = curr_volume > 2.0 * vol_ma_20_1d_aligned[i] if not np.isnan(vol_ma_20_1d_aligned[i]) else False
        
        # Donchian breakout conditions
        long_breakout = curr_close > high_20_aligned[i]
        short_breakout = curr_close < low_20_aligned[i]
        
        # Exit conditions: opposite breakout or regime change
        if position != 0:
            # Exit long: price breaks below lower Donchian or regime turns choppy
            if position == 1:
                if curr_close < low_20_aligned[i] or not trending_regime:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price breaks above upper Donchian or regime turns choppy
            elif position == -1:
                if curr_close > high_20_aligned[i] or not trending_regime:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Donchian breakout with volume confirmation and trending regime
        if position == 0 and trending_regime and volume_confirm:
            # Long: price breaks above upper Donchian
            if long_breakout:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian
            elif short_breakout:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_Breakout_1dVolumeSpike_ChopFilter_v1"
timeframe = "12h"
leverage = 1.0