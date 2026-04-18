#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray with weekly EMA50 regime and daily EMA13 filter.
# Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low) measures trend strength.
# Weekly EMA50 defines regime: only long when price > weekly EMA50 (bull), short when price < weekly EMA50 (bear).
# Daily EMA13 provides dynamic support/resistance for entries.
# Designed for low trade frequency (15-35/year) to minimize fee drag in 6h timeframe.
# Works in bull markets (trend following with regime) and bear markets (counter-trend reversals at EMA13).
name = "6h_ElderRay_1dEMA13_WeeklyEMA50_Regime"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get daily data for EMA13 (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    # Get weekly data for EMA50 regime (ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate daily EMA13 for Elder Ray
    ema_13_1d = pd.Series(df_1d['close']).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema_13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_13_1d)
    
    # Calculate weekly EMA50 for regime filter
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Elder Ray components: Bull Power = High - EMA13, Bear Power = EMA13 - Low
    bull_power = high - ema_13_1d_aligned
    bear_power = ema_13_1d_aligned - low
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for weekly EMA50 calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_13_1d_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(bull_power[i]) or
            np.isnan(bear_power[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: weekly EMA50 defines trend direction
        regime_bull = close[i] > ema_50_1w_aligned[i]  # Bull regime: price above weekly EMA50
        regime_bear = close[i] < ema_50_1w_aligned[i]  # Bear regime: price below weekly EMA50
        
        if position == 0:
            # Long: Bull power positive in bull regime (strong upward momentum)
            if regime_bull and bull_power[i] > 0:
                signals[i] = 0.25
                position = 1
            # Short: Bear power positive in bear regime (strong downward momentum)
            elif regime_bear and bear_power[i] > 0:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: bear power becomes positive (momentum shifts down)
            if bear_power[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: bull power becomes positive (momentum shifts up)
            if bull_power[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals