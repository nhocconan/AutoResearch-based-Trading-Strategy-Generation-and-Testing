#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray + 1d ADX Regime + Volume Confirmation
- Bull Power = High - EMA13(1d), Bear Power = EMA13(1d) - Low
- Long: Bull Power > 0 + Bear Power < 0 + 1d ADX > 25 (trending) + Volume > 1.5x 20-period avg
- Short: Bear Power > 0 + Bull Power < 0 + 1d ADX > 25 (trending) + Volume > 1.5x 20-period avg
- Exit: Opposite Elder Ray signal OR ADX < 20 (range) OR trailing stop (2.5x ATR)
- Uses 1d EMA13 as the core Elder Ray filter to avoid whipsaws
- ADX regime filter ensures we only trade strong trends, avoiding chop
- Volume confirmation reduces false signals
- Designed for both bull and bear markets: ADX > 25 works in any strong trend
- Target: 12-30 trades/year (50-120 total over 4 years) to minimize fee drag on 6h
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
    
    # Calculate ATR(14) for trailing stop
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with close
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Load 1d data ONCE before loop for Elder Ray and ADX
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate EMA13 on 1d close for Elder Ray
    ema_13_1d = pd.Series(df_1d['close']).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Bull Power and Bear Power
    bull_power = df_1d['high'].values - ema_13_1d
    bear_power = ema_13_1d - df_1d['low'].values
    
    # Calculate ADX(14) on 1d data
    # True Range
    tr1_1d = np.abs(df_1d['high'].values[1:] - df_1d['low'].values[1:])
    tr2_1d = np.abs(df_1d['high'].values[1:] - df_1d['close'].values[:-1])
    tr3_1d = np.abs(df_1d['low'].values[1:] - df_1d['close'].values[:-1])
    tr_1d = np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))
    tr_1d = np.concatenate([[np.nan], tr_1d])
    
    # Directional Movement
    dm_plus = np.where((df_1d['high'].values[1:] - df_1d['high'].values[:-1]) > 
                       (df_1d['low'].values[:-1] - df_1d['low'].values[1:]),
                       np.maximum(df_1d['high'].values[1:] - df_1d['high'].values[:-1], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    
    dm_minus = np.where((df_1d['low'].values[:-1] - df_1d['low'].values[1:]) > 
                        (df_1d['high'].values[1:] - df_1d['high'].values[:-1]),
                        np.maximum(df_1d['low'].values[:-1] - df_1d['low'].values[1:], 0), 0)
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed TR, DM+, DM-
    tr_14 = pd.Series(tr_1d).ewm(alpha=1/14, adjust=False).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    
    # Align HTF indicators to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    ema_13_aligned = align_htf_to_ltf(prices, df_1d, ema_13_1d)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    long_extreme = 0.0  # highest high since long entry
    short_extreme = 0.0  # lowest low since short entry
    
    # Start from index where all indicators are ready
    start_idx = max(20, 14, 13, 28)  # 20 for vol MA, 14 for ATR, 13 for EMA13, 28 for ADX (14+14)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(bull_power_aligned[i]) or 
            np.isnan(bear_power_aligned[i]) or 
            np.isnan(ema_13_aligned[i]) or 
            np.isnan(adx_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                long_extreme = 0.0
                short_extreme = 0.0
            continue
        
        # Elder Ray conditions
        bull_strong = bull_power_aligned[i] > 0
        bear_strong = bear_power_aligned[i] > 0
        
        # ADX regime filter: > 25 = trending, < 20 = range
        adx_trending = adx_aligned[i] > 25
        adx_ranging = adx_aligned[i] < 20
        
        # Volume spike confirmation
        volume_spike = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: Bull Power > 0, Bear Power < 0, ADX > 25 (trending), Volume spike
            if bull_strong and not bear_strong and adx_trending and volume_spike:
                signals[i] = 0.25
                position = 1
                long_extreme = high[i]
            # Short: Bear Power > 0, Bull Power < 0, ADX > 25 (trending), Volume spike
            elif bear_strong and not bull_strong and adx_trending and volume_spike:
                signals[i] = -0.25
                position = -1
                short_extreme = low[i]
        elif position == 1:
            # Update long extreme
            long_extreme = max(long_extreme, high[i])
            
            # Exit conditions:
            # 1. Price reverses 2.5x ATR from long extreme (trailing stop)
            # 2. Opposite Elder Ray signal (Bear Power > 0)
            # 3. ADX drops below 20 (range) - avoid whipsaws
            trailing_stop_long = close[i] < long_extreme - 2.5 * atr[i]
            opposite_signal = bear_power_aligned[i] > 0
            range_regime = adx_aligned[i] < 20
            
            if trailing_stop_long or opposite_signal or range_regime:
                signals[i] = 0.0
                position = 0
                long_extreme = 0.0
                short_extreme = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Update short extreme
            short_extreme = min(short_extreme, low[i])
            
            # Exit conditions:
            # 1. Price reverses 2.5x ATR from short extreme (trailing stop)
            # 2. Opposite Elder Ray signal (Bull Power > 0)
            # 3. ADX drops below 20 (range) - avoid whipsaws
            trailing_stop_short = close[i] > short_extreme + 2.5 * atr[i]
            opposite_signal = bull_power_aligned[i] > 0
            range_regime = adx_aligned[i] < 20
            
            if trailing_stop_short or opposite_signal or range_regime:
                signals[i] = 0.0
                position = 0
                long_extreme = 0.0
                short_extreme = 0.0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_1dADX_Regime_VolumeSpike_ATRStop"
timeframe = "6h"
leverage = 1.0