#!/usr/bin/env python3
"""
6h_ElderRay_Regime_Adaptive
Hypothesis: On 6h timeframe, Elder Ray Index (Bull Power/Bear Power) combined with a volatility regime filter (using ATR ratio) adapts to both bull and bear markets. In low volatility (chop), we mean-revert at extremes; in high volatility (trend), we follow the Elder Ray direction. Uses 1d HTF for regime and 6t for signals. Target: 50-150 total trades over 4 years (12-37/year) with discrete sizing to minimize fees.
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
    
    # Load 1d data ONCE before loop for HTF regime and EMA
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA21 for trend regime
    close_1d = df_1d['close'].values
    ema_21 = pd.Series(close_1d).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_aligned = align_htf_to_ltf(prices, df_1d, ema_21)
    
    # Calculate 1d ATR(14) for volatility regime
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
    tr2 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, tr2)
    tr = np.concatenate([[np.nan], tr])  # align length
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    # Avoid division by zero
    atr_ma_50 = pd.Series(atr_14).rolling(window=50, min_periods=50).mean().values
    atr_ratio = np.where(atr_ma_50 > 0, atr_14 / atr_ma_50, 1.0)
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # Calculate 6h Elder Ray components
    # Bull Power = High - EMA13
    # Bear Power = Low - EMA13
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # 6h EMA50 for dynamic reference (optional filter)
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need sufficient data for all indicators)
    start_idx = max(50, 21, 13)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_21_aligned[i]) or 
            np.isnan(atr_ratio_aligned[i]) or
            np.isnan(ema_13[i]) or
            np.isnan(ema_50[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Regime filters
        uptrend_regime = close[i] > ema_21_aligned[i]
        high_vol = atr_ratio_aligned[i] > 1.5  # ATR ratio > 1.5 = expanding volatility (trend)
        low_vol = atr_ratio_aligned[i] < 0.8   # ATR ratio < 0.8 = contracting volatility (chop)
        
        # Elder Ray signals
        strong_bull = bull_power[i] > 0 and bull_power[i] > np.abs(bear_power[i])
        strong_bear = bear_power[i] < 0 and np.abs(bear_power[i]) > bull_power[i]
        
        # Entry logic: adaptive to regime
        long_signal = False
        short_signal = False
        
        if high_vol:  # Trending regime: follow Elder Ray
            long_signal = strong_bull and close[i] > ema_50[i]
            short_signal = strong_bear and close[i] < ema_50[i]
        elif low_vol:  # Chop regime: mean revert at extremes
            # In chop, fade strong Elder Ray extremes
            long_signal = strong_bear and close[i] < ema_13[i]  # Bear power extreme -> long
            short_signal = strong_bull and close[i] > ema_13[i]  # Bull power extreme -> short
        else:  # Neutral regime: weak Elder Ray filter
            long_signal = bull_power[i] > 0 and close[i] > ema_50[i]
            short_signal = bear_power[i] < 0 and close[i] < ema_50[i]
        
        # Exit conditions
        exit_long = position == 1 and (not long_signal or bear_power[i] > 0)
        exit_short = position == -1 and (not short_signal or bull_power[i] < 0)
        
        if long_signal and position != 1:
            signals[i] = 0.25
            position = 1
        elif exit_long:
            signals[i] = 0.0
            position = 0
        elif short_signal and position != -1:
            signals[i] = -0.25
            position = -1
        elif exit_short:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_Regime_Adaptive"
timeframe = "6h"
leverage = 1.0