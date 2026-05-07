#!/usr/bin/env python3
name = "6h_ElderRay_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

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
    
    # Load 1d data ONCE for trend filter and Elder Ray calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1d EMA13 for Elder Ray calculation
    ema_13_1d = pd.Series(df_1d['close']).ewm(span=13, adjust=False, min_periods=13).mean().values
    # 1d EMA13 for trend filter (same as Elder Ray base)
    ema_13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_13_1d)
    
    # Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power_1d = df_1d['high'].values - ema_13_1d
    bear_power_1d = df_1d['low'].values - ema_13_1d
    
    # Align Bull/Bear Power to 6t
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # Volume spike detection on 6h
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 13)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_13_1d_aligned[i]) or np.isnan(bull_power_aligned[i]) or 
            np.isnan(bear_power_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bull Power > 0 (bullish momentum) + price above EMA13 + volume spike
            if bull_power_aligned[i] > 0 and close[i] > ema_13_1d_aligned[i] and volume[i] > vol_ma_20[i] * 1.5:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 (bearish momentum) + price below EMA13 + volume spike
            elif bear_power_aligned[i] < 0 and close[i] < ema_13_1d_aligned[i] and volume[i] > vol_ma_20[i] * 1.5:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Bull Power turns negative or price below EMA13
            if bull_power_aligned[i] <= 0 or close[i] < ema_13_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Bear Power turns positive or price above EMA13
            if bear_power_aligned[i] >= 0 or close[i] > ema_13_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d trend filter and volume confirmation
# - Bull Power (High - EMA13) > 0 indicates bullish momentum, Bear Power (Low - EMA13) < 0 indicates bearish momentum
# - Entry when power aligns with price position relative to EMA13 and volume confirms (1.5x average)
# - Uses 1d Elder Ray aligned to 6h to avoid look-ahead, ensuring only completed 1d bar data is used
# - Works in both bull (buy when Bull Power > 0) and bear (sell when Bear Power < 0) markets
# - Volume filter reduces false signals during low-activity periods
# - Exit when power signal invalidates or price crosses EMA13
# - Position size 0.25 balances return and risk, targeting ~50-150 trades over 4 years
# - Novel combination: Elder Ray momentum + volume spike on 6h with 1d alignment
# - Avoids saturated Donchian/Camarilla families while capturing institutional order flow via power metrics