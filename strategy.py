#!/usr/bin/env python3
"""
6h_ElderRay_BullBearPower_1dTrend_RegimeFilter_v1
Hypothesis: Elder Ray (Bull Power/Bear Power) combined with 1d EMA50 trend and Bollinger Bandwidth regime filter to capture momentum continuations in both trending and ranging markets. Bull Power > 0 indicates buying strength, Bear Power < 0 indicates selling strength. Regime filter avoids whipsaws in high volatility. Target: 60-120 total trades over 4 years (15-30/year).
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
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Calculate Bollinger Bandwidth on 1d for regime filter (low BW = low volatility)
    bb_period = 20
    bb_std = 2.0
    sma_1d = pd.Series(close_1d).rolling(window=bb_period, min_periods=bb_period).mean().values
    std_1d = pd.Series(close_1d).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_bb = sma_1d + (bb_std * std_1d)
    lower_bb = sma_1d - (bb_std * std_1d)
    bb_width = (upper_bb - lower_bb) / sma_1d  # Normalized bandwidth
    bb_width_aligned = align_htf_to_ltf(prices, df_1d, bb_width)
    
    # Calculate 6h EMA13 for Elder Ray (EMA of close)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need sufficient data for all indicators)
    start_idx = max(50, bb_period, 13)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_aligned[i]) or 
            np.isnan(bb_width_aligned[i]) or
            np.isnan(bull_power[i]) or
            np.isnan(bear_power[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Regime filter: avoid extremely low volatility (BB width too tight) and high volatility
        # Low BB width indicates squeeze/low vol, high BB width indicates expansion/high vol
        # We want moderate volatility regime: BB width between 0.01 and 0.05 (adjust based on asset)
        vol_regime = (bb_width_aligned[i] > 0.01) & (bb_width_aligned[i] < 0.05)
        
        # 1d trend filter (EMA50)
        uptrend = close[i] > ema_50_aligned[i]
        downtrend = close[i] < ema_50_aligned[i]
        
        # Long logic: Bull Power > 0 (buying strength) + uptrend + moderate volatility
        if bull_power[i] > 0 and uptrend and vol_regime:
            if position != 1:
                signals[i] = 0.25
                position = 1
            else:
                signals[i] = 0.25
        # Short logic: Bear Power < 0 (selling strength) + downtrend + moderate volatility
        elif bear_power[i] < 0 and downtrend and vol_regime:
            if position != -1:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = -0.25
        # Exit conditions: power reverses or trend weakens or volatility regime changes
        elif position == 1 and (bull_power[i] <= 0 or not uptrend or not vol_regime):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (bear_power[i] >= 0 or not downtrend or not vol_regime):
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

name = "6h_ElderRay_BullBearPower_1dTrend_RegimeFilter_v1"
timeframe = "6h"
leverage = 1.0