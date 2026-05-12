#!/usr/bin/env python3
"""
6h_Elder_Ray_Regime_Adaptive
Hypothesis: Use Elder Ray (Bull/Bear Power) from 1d to determine market regime, then apply adaptive entries:
- In bull regime (Bull Power > 0 and rising): buy pullbacks to EMA21 on 6s
- In bear regime (Bear Power < 0 and falling): sell rallies to EMA21 on 6s
- In range regime (|Bull Power| < EMA13 of |Bear Power|): fade at Bollinger Bands (20,2)
This adapts to trending and ranging markets, reducing whipsaws. Targets 15-30 trades/year.
"""

name = "6h_Elder_Ray_Regime_Adaptive"
timeframe = "6h"
leverage = 1.0

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

    # Get 1d data for Elder Ray and regime detection ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)

    # Calculate EMA13 and EMA21 on 1d close for Elder Ray
    close_1d = df_1d['close'].values
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema21_1d = pd.Series(close_1d).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = df_1d['high'].values - ema13_1d
    bear_power = df_1d['low'].values - ema13_1d
    
    # Smooth Elder Ray with EMA13 for better signals
    bull_power_smooth = pd.Series(bull_power).ewm(span=13, adjust=False, min_periods=13).mean().values
    bear_power_smooth = pd.Series(bear_power).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Regime detection: 
    # Bull regime: Bull Power > 0 and rising (current > previous)
    # Bear regime: Bear Power < 0 and falling (current < previous)
    # Range regime: otherwise
    bull_regime = (bull_power_smooth > 0) & (np.roll(bull_power_smooth, 1) <= bull_power_smooth)
    bear_regime = (bear_power_smooth < 0) & (np.roll(bear_power_smooth, 1) >= bear_power_smooth)
    range_regime = ~(bull_regime | bear_regime)
    
    # Align regime signals to 6h
    bull_regime_aligned = align_htf_to_ltf(prices, df_1d, bull_regime.astype(float))
    bear_regime_aligned = align_htf_to_ltf(prices, df_1d, bear_regime.astype(float))
    range_regime_aligned = align_htf_to_ltf(prices, df_1d, range_regime.astype(float))

    # Get Bollinger Bands from 1d for range regime entries
    sma20_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std20_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_bb = sma20_1d + (2 * std20_1d)
    lower_bb = sma20_1d - (2 * std20_1d)
    upper_bb_aligned = align_htf_to_ltf(prices, df_1d, upper_bb)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1d, lower_bb)

    # Get EMA21 from 6s for trend regime entries
    ema21_6s = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values

    # Volume confirmation: current volume > 1.2x average of last 10 periods
    vol_ma = pd.Series(volume).rolling(window=10, min_periods=10).mean().values
    volume_ok = volume > (1.2 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(30, n):  # Start after warmup
        if (np.isnan(bull_regime_aligned[i]) or np.isnan(bear_regime_aligned[i]) or 
            np.isnan(range_regime_aligned[i]) or np.isnan(upper_bb_aligned[i]) or
            np.isnan(lower_bb_aligned[i]) or np.isnan(ema21_6s[i]) or
            np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # BULL REGIME: Buy pullbacks to EMA21
            if bull_regime_aligned[i] > 0.5 and close[i] <= ema21_6s[i] * 1.005 and volume_ok[i]:
                signals[i] = 0.25
                position = 1
            # BEAR REGIME: Sell rallies to EMA21
            elif bear_regime_aligned[i] > 0.5 and close[i] >= ema21_6s[i] * 0.995 and volume_ok[i]:
                signals[i] = -0.25
                position = -1
            # RANGE REGIME: Fade at Bollinger Bands
            elif range_regime_aligned[i] > 0.5:
                if close[i] <= lower_bb_aligned[i] and volume_ok[i]:
                    signals[i] = 0.25
                    position = 1
                elif close[i] >= upper_bb_aligned[i] and volume_ok[i]:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: 
            # - Bull regime ends OR price crosses above EMA21 (take profit)
            # - Bear regime starts (reverse)
            # - Range regime and price at upper BB (take profit)
            if (bull_regime_aligned[i] <= 0.5 and bear_regime_aligned[i] <= 0.5) or \
               close[i] >= ema21_6s[i] * 1.01 or \
               (range_regime_aligned[i] > 0.5 and close[i] >= upper_bb_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: 
            # - Bear regime ends OR price crosses below EMA21 (take profit)
            # - Bull regime starts (reverse)
            # - Range regime and price at lower BB (take profit)
            if (bear_regime_aligned[i] <= 0.5 and bull_regime_aligned[i] <= 0.5) or \
               close[i] <= ema21_6s[i] * 0.99 or \
               (range_regime_aligned[i] > 0.5 and close[i] <= lower_bb_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals