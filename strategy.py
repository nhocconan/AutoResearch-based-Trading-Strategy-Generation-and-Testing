#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d regime filter
# Elder Ray measures bull/bear power relative to EMA13. Combined with 1d regime:
# - Bull regime: price > 1d EMA50 -> only take long signals when bull power > 0
# - Bear regime: price < 1d EMA50 -> only take short signals when bear power < 0
# This avoids counter-trend trades and works in both bull/bear markets by adapting.
# Target: 50-150 total trades over 4 years (~12-37/year) with size 0.25.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Load 1d data for regime filter (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # 1d EMA(50) for regime filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Elder Ray components on 6h data
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13  # Bull Power = High - EMA13
    bear_power = low - ema13   # Bear Power = Low - EMA13
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(ema13[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine regime from 1d EMA50
        bull_regime = close[i] > ema_50_1d_aligned[i]
        bear_regime = close[i] < ema_50_1d_aligned[i]
        
        if position == 0:
            # Long: bull regime AND bull power positive (buying strength)
            if bull_regime and bull_power[i] > 0:
                signals[i] = 0.25
                position = 1
            # Short: bear regime AND bear power negative (selling pressure)
            elif bear_regime and bear_power[i] < 0:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: regime change or power signal reversal
            if position == 1:
                # Exit long: bear regime OR bull power turns negative
                if bear_regime or bull_power[i] <= 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short: bull regime OR bear power turns positive
                if bull_regime or bear_power[i] >= 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_1dEMA50_Regime"
timeframe = "6h"
leverage = 1.0