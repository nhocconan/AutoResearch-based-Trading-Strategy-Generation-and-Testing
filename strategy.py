#!/usr/bin/env python3
"""
6h_ElderRay_Regime_VolumeFilter
Hypothesis: Elder Ray (Bull/Bear Power) with 1d regime filter (ADX>25 = trending, ADX<20 = range) and volume confirmation.
In trending markets: go long when Bear Power < 0 and rising, short when Bull Power > 0 and falling.
In ranging markets: fade extremes (long when Bull Power < -std, short when Bear Power > std).
Volume spike (>1.5x 20 MA) confirms entry. Discrete sizing 0.25. Target: 60-120 total trades over 4 years (15-30/year).
Works in both bull (trend following) and bear (mean reversion in ranges) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for Elder Ray and regime)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # === 1d EMA13 for Elder Ray calculation ===
    ema13_1d = pd.Series(df_1d['close'].values).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # === Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13 ===
    bull_power_1d = df_1d['high'].values - ema13_1d
    bear_power_1d = df_1d['low'].values - ema13_1d
    
    # Align Elder Ray to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # === 1d ADX for regime detection (trending vs ranging) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    
    # Smooth DM and TR
    dm_plus_smooth = pd.Series(dm_plus).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(span=14, adjust=False, min_periods=14).mean().values
    tr_smooth = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / tr_smooth
    di_minus = 100 * dm_minus_smooth / tr_smooth
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx_1d = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # === 6h ATR (14-period) for stoploss ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === Volume spike filter (1.5x 20-period MA) ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Precompute power statistics for regime-based thresholds
    bull_power_valid = bull_power_aligned[~np.isnan(bull_power_aligned)]
    bear_power_valid = bear_power_aligned[~np.isnan(bear_power_aligned)]
    if len(bull_power_valid) > 50:
        bull_power_std = np.std(bull_power_valid)
        bear_power_std = np.std(bear_power_valid)
    else:
        bull_power_std = bear_power_std = 1.0  # fallback
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) 
            or np.isnan(adx_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        volume_now = volume[i]
        bull_power = bull_power_aligned[i]
        bear_power = bear_power_aligned[i]
        adx = adx_aligned[i]
        vol_avg = vol_ma[i]
        
        # Volume spike: current volume > 1.5x average
        volume_spike = volume_now > 1.5 * vol_avg
        
        if position == 0:
            # Regime-based entry logic
            if adx > 25:  # Trending regime
                # Long: Bear Power negative and rising (less negative)
                # Short: Bull Power positive and falling (less positive)
                if i >= 101:
                    bear_power_prev = bear_power_aligned[i-1]
                    bull_power_prev = bull_power_aligned[i-1]
                    long_condition = (bear_power < 0) and (bear_power > bear_power_prev) and volume_spike
                    short_condition = (bull_power > 0) and (bull_power < bull_power_prev) and volume_spike
                else:
                    long_condition = (bear_power < 0) and volume_spike
                    short_condition = (bull_power > 0) and volume_spike
            else:  # Ranging regime (ADX <= 25)
                # Fade extremes: long when Bull Power very weak, short when Bear Power very strong
                long_condition = (bull_power < -0.5 * bull_power_std) and volume_spike
                short_condition = (bear_power > 0.5 * bear_power_std) and volume_spike
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = price
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Check stoploss (2.0x ATR)
            if price < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Regime/exit logic
            elif adx > 25:  # Trending: exit when Bear Power turns positive
                if bear_power > 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # Ranging: exit when Bull Power recovers
                if bull_power > -0.2 * bull_power_std:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
        
        elif position == -1:
            # Check stoploss (2.0x ATR)
            if price > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Regime/exit logic
            elif adx > 25:  # Trending: exit when Bull Power turns negative
                if bull_power < 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:  # Ranging: exit when Bear Power recovers
                if bear_power < 0.2 * bear_power_std:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_Regime_VolumeFilter"
timeframe = "6h"
leverage = 1.0