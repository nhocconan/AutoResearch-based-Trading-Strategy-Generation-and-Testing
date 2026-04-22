#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Elder Ray + 1w EMA trend filter
    # Elder Ray uses 13-period EMA to measure bull/bear power:
    # Bull Power = High - EMA13, Bear Power = Low - EMA13
    # We go long when Bull Power > 0 and weekly trend is up (price > weekly EMA34)
    # We go short when Bear Power < 0 and weekly trend is down (price < weekly EMA34)
    # This combines momentum (EMA) with price action extremes (high/low) to capture
    # strong moves while avoiding chop. Weekly EMA34 provides strong trend filter.
    # Target: 15-25 trades/year per symbol (60-100 total over 4 years).
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate 6h EMA13 for Elder Ray
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components
    bull_power = high - ema13  # Strength of bulls: ability to push price above EMA
    bear_power = low - ema13   # Strength of bears: ability to push price below EMA
    
    # Load 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(13, n):  # Start after EMA warmup
        # Skip if weekly trend data not ready
        if np.isnan(ema34_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bull Power positive AND price above weekly EMA34 (uptrend)
            if bull_power[i] > 0 and close[i] > ema34_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power negative AND price below weekly EMA34 (downtrend)
            elif bear_power[i] < 0 and close[i] < ema34_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Elder Ray signal reverses OR price crosses weekly EMA34
            if position == 1:
                if bull_power[i] <= 0 or close[i] < ema34_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if bear_power[i] >= 0 or close[i] > ema34_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_1wEMA34_TrendFilter_v1"
timeframe = "6h"
leverage = 1.0