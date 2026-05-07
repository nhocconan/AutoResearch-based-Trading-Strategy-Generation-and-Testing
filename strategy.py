#!/usr/bin/env python3
"""
6h_ElderRay_BullBearPower_1dTrend
Hypothesis: Elder Ray Bull Power (high - EMA13) and Bear Power (EMA13 - low) combined with 1d EMA34 trend filter captures institutional accumulation/distribution.
Long when Bull Power > 0 and Bear Power < 0 with 1d uptrend. Short when Bear Power > 0 and Bull Power < 0 with 1d downtrend.
Uses 13-period EMA for Ray calculation, works in bull/bear by following higher timeframe trend.
Target: 50-120 total trades over 4 years (12-30/year) to avoid fee drag.
"""

name = "6h_ElderRay_BullBearPower_1dTrend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate EMA13 for Elder Ray (using close)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema13  # High minus EMA13
    bear_power = ema13 - low   # EMA13 minus Low
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need sufficient warmup for EMA13
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if np.isnan(ema_34_1d_aligned[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bull Power > 0 (strong buying) AND Bear Power < 0 (weak selling) AND 1d uptrend
            if bull_power[i] > 0 and bear_power[i] < 0 and ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power > 0 (strong selling) AND Bull Power < 0 (weak buying) AND 1d downtrend
            elif bear_power[i] > 0 and bull_power[i] < 0 and ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1]:
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: when the power signals weaken or reverse
            if position == 1:
                # Exit long when Bull Power turns negative or Bear Power turns positive
                if bull_power[i] <= 0 or bear_power[i] >= 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short when Bear Power turns negative or Bull Power turns positive
                if bear_power[i] <= 0 or bull_power[i] >= 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals