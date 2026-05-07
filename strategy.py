#!/usr/bin/env python3
name = "6h_ElderRay_BullBearPower_1dTrend"
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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA13 for trend filter (shorter for responsiveness)
    ema13_1d = pd.Series(df_1d['close']).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema13_1d)
    
    # 13-period EMA for Elder Ray calculation (on 6h data)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13
    bull_power = high - ema13
    # Bear Power = Low - EMA13
    bear_power = low - ema13
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 13  # Wait for EMA13
    
    for i in range(start_idx, n):
        if np.isnan(ema13_1d_aligned[i]) or np.isnan(ema13[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bull Power > 0 (buyers strong) AND 1d trend up (price > EMA13)
            if bull_power[i] > 0 and close[i] > ema13_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 (sellers strong) AND 1d trend down (price < EMA13)
            elif bear_power[i] < 0 and close[i] < ema13_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Bull Power <= 0 (buyers weakening) OR 1d trend turns down
            if bull_power[i] <= 0 or close[i] < ema13_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Bear Power >= 0 (sellers weakening) OR 1d trend turns up
            if bear_power[i] >= 0 or close[i] > ema13_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Elder Ray (Bull/Bear Power) measures buyer/seller strength relative to 13-period EMA.
# Long when Bull Power > 0 (buyers pushing above EMA) AND 1d trend up (price > 1d EMA13).
# Short when Bear Power < 0 (sellers pushing below EMA) AND 1d trend down (price < 1d EMA13).
# Uses 13-period EMA for responsiveness. Trend filter from higher timeframe (1d) reduces whipsaws.
# Works in bull markets (buy strength + uptrend) and bear markets (sell strength + downtrend).
# Target: 20-40 trades/year to minimize fee decay while capturing sustained directional moves.