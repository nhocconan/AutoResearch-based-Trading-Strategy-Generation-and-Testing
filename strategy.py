#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Power + Daily Trend Filter
# Elder Ray measures bull/bear power via EMA13: Bull Power = High - EMA13, Bear Power = Low - EMA13
# Trend filter: 1d EMA50 slope (rising/falling) to align with higher timeframe trend
# Long when Bull Power > 0 and 1d EMA50 rising, Short when Bear Power < 0 and 1d EMA50 falling
# Works in bull markets (captures strength) and bear markets (captures weakness)
# Low turnover expected: ~15-30 trades/year per symbol

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Load 1d data ONCE for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA50 and its slope
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_slope = np.diff(ema50_1d, prepend=np.nan)
    
    # Align 1d EMA50 slope to 6h timeframe
    ema50_slope_aligned = align_htf_to_ltf(prices, df_1d, ema50_slope)
    
    # Calculate EMA13 for Elder Ray (6 timeframe)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema13  # Higher values = stronger bullish pressure
    bear_power = low - ema13   # Lower values = stronger bearish pressure
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(50, 13)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema50_slope_aligned[i]) or 
            np.isnan(ema13[i]) or
            np.isnan(bull_power[i]) or
            np.isnan(bear_power[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: 1d EMA50 slope indicates trend direction
        ema50_rising = ema50_slope_aligned[i] > 0
        ema50_falling = ema50_slope_aligned[i] < 0
        
        if position == 0:
            # Enter long: Bull Power positive + 1d EMA50 rising
            if (bull_power[i] > 0 and ema50_rising):
                position = 1
                signals[i] = position_size
            # Enter short: Bear Power negative + 1d EMA50 falling
            elif (bear_power[i] < 0 and ema50_falling):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Bull Power turns negative OR trend turns bearish
            if (bull_power[i] <= 0 or not ema50_rising):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Bear Power turns positive OR trend turns bullish
            if (bear_power[i] >= 0 or not ema50_falling):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_ElderRay_Power_Trend_v1"
timeframe = "6h"
leverage = 1.0