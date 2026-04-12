#!/usr/bin/env python3
"""
6h_1d_1w_Momentum_Regime_v1
Hypothesis: On 6b timeframe, combine 1d momentum (EMA crossover) with 1w regime filter (price above/below weekly EMA200) and volume confirmation.
In bull regime (price > weekly EMA200), take long signals from 1d EMA crossover; in bear regime (price < weekly EMA200), take short signals.
Uses volume spike to confirm momentum strength. Designed to work in both bull and bear markets by adapting direction based on higher timeframe trend.
Target: 20-40 trades per year (80-160 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_1w_Momentum_Regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1D data for momentum signals
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Get 1W data for regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    daily_close = df_1d['close'].values
    weekly_close = df_1w['close'].values
    
    # === 1D EMA CROSSOVER (fast=9, slow=21) ===
    ema_fast = pd.Series(daily_close).ewm(span=9, adjust=False, min_periods=9).mean().values
    ema_slow = pd.Series(daily_close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Bullish crossover: fast crosses above slow
    bullish_cross = (ema_fast > ema_slow) & (np.roll(ema_fast, 1) <= np.roll(ema_slow, 1))
    # Bearish crossover: fast crosses below slow
    bearish_cross = (ema_fast < ema_slow) & (np.roll(ema_fast, 1) >= np.roll(ema_slow, 1))
    
    # Align crossovers to 6h timeframe
    bullish_cross_6h = align_htf_to_ltf(prices, df_1d, bullish_cross.astype(float))
    bearish_cross_6h = align_htf_to_ltf(prices, df_1d, bearish_cross.astype(float))
    
    # === 1W REGIME FILTER (price vs EMA200) ===
    ema200 = pd.Series(weekly_close).ewm(span=200, adjust=False, min_periods=200).mean().values
    bull_regime = weekly_close > ema200  # Bull regime: price above weekly EMA200
    bear_regime = weekly_close < ema200  # Bear regime: price below weekly EMA200
    
    # Align regime to 6h timeframe
    bull_regime_6h = align_htf_to_ltf(prices, df_1w, bull_regime.astype(float))
    bear_regime_6h = align_htf_to_ltf(prices, df_1w, bear_regime.astype(float))
    
    # === VOLUME SPIKE (2x 20-period average on 6h) ===
    vol_ma = np.full(n, np.nan)
    if n >= 20:
        vol_sum = np.sum(volume[:20])
        vol_ma[19] = vol_sum / 20
        for i in range(20, n):
            vol_sum = vol_sum - volume[i-20] + volume[i]
            vol_ma[i] = vol_sum / 20
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any data invalid
        if (np.isnan(bullish_cross_6h[i]) or np.isnan(bearish_cross_6h[i]) or
            np.isnan(bull_regime_6h[i]) or np.isnan(bear_regime_6h[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Entry conditions: crossover in direction of regime + volume confirmation
        long_entry = bool(bullish_cross_6h[i]) and bool(bull_regime_6h[i]) and vol_spike[i]
        short_entry = bool(bearish_cross_6h[i]) and bool(bear_regime_6h[i]) and vol_spike[i]
        
        # Exit conditions: opposite crossover or regime change
        long_exit = bool(bearish_cross_6h[i]) or not bool(bull_regime_6h[i])
        short_exit = bool(bullish_cross_6h[i]) or not bool(bear_regime_6h[i])
        
        # Signal logic
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals