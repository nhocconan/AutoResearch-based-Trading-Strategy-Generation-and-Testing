#!/usr/bin/env python3
"""
6h_ElderRay_BullBearPower_1dTrend_RegimeFilter
Hypothesis: Elder Ray (Bull Power = High - EMA13, Bear Power = Low - EMA13) identifies 6h momentum exhaustion. 
In 1d uptrend (price > EMA50), look for Bull Power turning negative after positive (long exhaustion) to short.
In 1d downtrend (price < EMA50), look for Bear Power turning positive after negative (short exhaustion) to long.
Adds 1d choppiness filter: only trade when CHOP(14) < 61.8 (trending regime) to avoid whipsaws in ranging markets.
Target: 12-30 trades/year per symbol. Works in bull markets by catching trend exhaustion longs, in bear markets by catching trend exhaustion shorts.
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
    
    # 1d data for trend filter and choppiness regime (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA50 for trend filter (loaded ONCE)
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # 1d choppiness index: CHOP = 100 * log10(sum(ATR(14)) / log10(highest_high - lowest_low) / log10(14))
    # Simplified: CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    tr_1d = np.maximum(df_1d['high'].values - df_1d['low'].values,
                       np.maximum(np.abs(df_1d['high'].values - np.concatenate([[df_1d['close'].values[0]], df_1d['close'].values[:-1]])),
                                  np.abs(df_1d['low'].values - np.concatenate([[df_1d['close'].values[0]], df_1d['close'].values[:-1]]))))
    atr_14_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    sum_atr_14 = pd.Series(atr_14_1d).rolling(window=14, min_periods=14).sum().values
    highest_high_14 = pd.Series(df_1d['high'].values).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(df_1d['low'].values).rolling(window=14, min_periods=14).min().values
    chop_denom = np.log10(highest_high_14 - lowest_low_14) * np.log10(14)
    chop_denom = np.where(chop_denom == 0, 1e-10, chop_denom)  # avoid div by zero
    chop_1d = 100 * (np.log10(sum_atr_14) / chop_denom)
    
    # Align HTF indicators to LTF (6h)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # 6h EMA13 for Elder Ray calculation
    ema_13_6h = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema_13_6h
    bear_power = low - ema_13_6h
    
    # Bull Power turning negative after being positive (exhaustion signal for longs -> short)
    # Bear Power turning positive after being negative (exhaustion signal for shorts -> long)
    bull_power_prev = np.roll(bull_power, 1)
    bear_power_prev = np.roll(bear_power, 1)
    bull_power_prev[0] = 0
    bear_power_prev[0] = 0
    
    bull_exhaustion = (bull_power < 0) & (bull_power_prev > 0)  # bull power just turned negative
    bear_exhaustion = (bear_power > 0) & (bear_power_prev < 0)  # bear power just turned positive
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need EMA13 (13), plus any warmup from HTF alignment
    start_idx = max(50, 13)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(chop_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade in trending markets (CHOP < 61.8)
        if chop_1d_aligned[i] >= 61.8:
            # In ranging markets, stay flat or reduce position
            signals[i] = 0.0
            position = 0
            continue
        
        if position == 0:
            # In 1d uptrend, bull exhaustion -> short
            # In 1d downtrend, bear exhaustion -> long
            if (close[i] > ema_50_1d_aligned[i]) and bull_exhaustion[i]:
                signals[i] = -0.25  # short 25%
                position = -1
            elif (close[i] < ema_50_1d_aligned[i]) and bear_exhaustion[i]:
                signals[i] = 0.25   # long 25%
                position = 1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold until bear exhaustion (short signal) or trend change
            signals[i] = 0.25
            # Exit: bear exhaustion OR trend turns down (price < EMA50)
            if bear_exhaustion[i] or (close[i] < ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold until bull exhaustion (long signal) or trend change
            signals[i] = -0.25
            # Exit: bull exhaustion OR trend turns up (price > EMA50)
            if bull_exhaustion[i] or (close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_ElderRay_BullBearPower_1dTrend_RegimeFilter"
timeframe = "6h"
leverage = 1.0