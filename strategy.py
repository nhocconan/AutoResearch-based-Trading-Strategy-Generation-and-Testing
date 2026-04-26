#!/usr/bin/env python3
"""
6h_ElderRay_BullBearPower_Regime_v1
Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d trend regime filter and volume confirmation.
Long when Bull Power > 0 AND Bear Power < 0 AND 1d uptrend AND volume > 1.5x MA.
Short when Bear Power < 0 AND Bull Power > 0 AND 1d downtrend AND volume > 1.5x MA.
Elder Ray measures bull/bear strength relative to EMA13, helping identify regime-appropriate entries.
This avoids whipsaws by requiring alignment with higher timeframe trend and volume confirmation.
Target: 50-150 trades over 4 years (12-37/year) with discrete sizing (0.25) to minimize fee drag.
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
    
    # Load 1d data ONCE before loop for trend filter and EMA13
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA34 for trend filter (regime)
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    # Trend: 1 = uptrend (close > EMA34), -1 = downtrend (close < EMA34), 0 = invalid
    trend_1d = np.where(ema_34_1d_aligned > 0, 
                        np.where(close > ema_34_1d_aligned, 1, -1), 
                        0)
    
    # Calculate 6h EMA13 for Elder Ray (need enough lookback)
    ema_13 = pd.Series(close).ewm(span=13, min_periods=13, adjust=False).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Volume filter: volume > 1.5 * volume_ma(20)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 34 for 1d EMA, 20 for volume MA, 13 for 6h EMA)
    start_idx = max(34, 20, 13)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(ema_13[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(trend_1d[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Entry conditions
        if position == 0:
            # Long: Bull Power > 0 AND Bear Power < 0 AND 1d uptrend AND volume spike
            if bull_power[i] > 0 and bear_power[i] < 0 and trend_1d[i] == 1 and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 AND Bull Power > 0 AND 1d downtrend AND volume spike
            elif bear_power[i] < 0 and bull_power[i] > 0 and trend_1d[i] == -1 and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: Bull Power <= 0 OR Bear Power >= 0 OR 1d trend turns down
            if bull_power[i] <= 0 or bear_power[i] >= 0 or trend_1d[i] == -1:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: Bear Power >= 0 OR Bull Power <= 0 OR 1d trend turns up
            if bear_power[i] >= 0 or bull_power[i] <= 0 or trend_1d[i] == 1:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_ElderRay_BullBearPower_Regime_v1"
timeframe = "6h"
leverage = 1.0