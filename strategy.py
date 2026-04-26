#!/usr/bin/env python3
"""
6h_ElderRay_BullBearPower_1dTrendFilter_v1
Hypothesis: Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low) with 1d EMA34 trend filter on 6h timeframe.
Long when Bull Power > 0 and Bear Power < 0 (bullish momentum) and price > 1d EMA34 (uptrend).
Short when Bull Power < 0 and Bear Power > 0 (bearish momentum) and price < 1d EMA34 (downtrend).
Uses discrete position sizing (0.0, ±0.25) to minimize fee churn. Targets 50-150 total trades over 4 years.
Works in bull/bear: 1d EMA34 trend filter adapts to market direction, Elder Ray filters counter-trend noise.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # EMA13 for Elder Ray (6h timeframe)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema_13  # Bull Power = High - EMA13
    bear_power = ema_13 - low   # Bear Power = EMA13 - Low
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of 1d EMA (34), EMA13 (13)
    start_idx = max(34, 13)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if np.isnan(ema_34_1d_aligned[i]) or np.isnan(ema_13[i]):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        ema_34_1d_val = ema_34_1d_aligned[i]
        ema_13_val = ema_13[i]
        close_val = close[i]
        bull_val = bull_power[i]
        bear_val = bear_power[i]
        
        if position == 0:
            # Long: Bull Power > 0 AND Bear Power < 0 (bullish momentum) AND price > 1d EMA34 (uptrend)
            long_signal = (bull_val > 0) and (bear_val < 0) and (close_val > ema_34_1d_val)
            # Short: Bull Power < 0 AND Bear Power > 0 (bearish momentum) AND price < 1d EMA34 (downtrend)
            short_signal = (bull_val < 0) and (bear_val > 0) and (close_val < ema_34_1d_val)
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: momentum deterioration or trend reversal
            if (bull_val <= 0) or (bear_val >= 0) or (close_val < ema_34_1d_val):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: momentum deterioration or trend reversal
            if (bull_val >= 0) or (bear_val <= 0) or (close_val > ema_34_1d_val):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_ElderRay_BullBearPower_1dTrendFilter_v1"
timeframe = "6h"
leverage = 1.0