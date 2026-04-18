#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d EMA13 and weekly EMA50 regime filter.
# Bull Power = High - EMA13, Bear Power = Low - EMA13 (daily timeframe).
# Weekly EMA50 defines trend: above = bull regime (long bias), below = bear regime (short bias).
# Entry: Bull Power > 0 in bull regime for long, Bear Power < 0 in bear regime for short.
# Exit: Opposite power signal or power crosses zero.
# Designed for low trade frequency (~20-40/year) to minimize fee drag in 6h timeframe.
# Works in bull markets (buy strength in uptrend) and bear markets (sell weakness in downtrend).
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
    
    # Get daily data for Elder Ray calculation (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    # Get weekly data for EMA50 regime filter (ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate daily EMA13 for Elder Ray
    ema_13_1d = pd.Series(df_1d['close']).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Bull Power (High - EMA13) and Bear Power (Low - EMA13)
    bull_power = df_1d['high'].values - ema_13_1d
    bear_power = df_1d['low'].values - ema_13_1d
    
    # Align to 6s timeframe (wait for daily bar to close)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Calculate weekly EMA50 for regime filter
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 13  # Wait for EMA13 calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(bull_power_aligned[i]) or
            np.isnan(bear_power_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime: weekly EMA50 slope (bull if rising, bear if falling)
        if i >= 1:
            ema_50_prev = ema_50_1w_aligned[i-1]
            ema_50_curr = ema_50_1w_aligned[i]
            bull_regime = ema_50_curr > ema_50_prev  # Rising EMA50 = bull regime
            bear_regime = ema_50_curr < ema_50_prev  # Falling EMA50 = bear regime
        else:
            bull_regime = False
            bear_regime = False
        
        if position == 0:
            # Long: Bull Power > 0 in bull regime
            if bull_regime and bull_power_aligned[i] > 0:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 in bear regime
            elif bear_regime and bear_power_aligned[i] < 0:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Bear Power >= 0 (loss of bearish pressure) or Bull Power <= 0
            if bear_power_aligned[i] >= 0 or bull_power_aligned[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Bull Power <= 0 (loss of bullish pressure) or Bear Power >= 0
            if bull_power_aligned[i] <= 0 or bear_power_aligned[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals