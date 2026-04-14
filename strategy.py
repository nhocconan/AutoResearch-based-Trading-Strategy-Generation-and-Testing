#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Williams Alligator + Elder Ray with 12h trend filter
# Long when: Alligator jaws < teeth < lips (bullish alignment) AND Bull Power > 0 AND price > 12h EMA50
# Short when: Alligator jaws > teeth > lips (bearish alignment) AND Bear Power < 0 AND price < 12h EMA50
# Exit when Alligator alignment breaks (jaws > teeth OR teeth < lips for long; inverse for short)
# Uses Williams Alligator (SMAs of median price) and Elder Ray (EMA13-based power) for trend strength
# 12h EMA50 filter ensures alignment with higher timeframe trend
# Target: 50-150 total trades over 4 years (12-37/year) to balance opportunity and cost

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate median price for Alligator
    median_price = (high + low) / 2.0
    
    # Williams Alligator: SMAs of median price
    jaws = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values  # 13-period
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values    # 8-period
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values     # 5-period
    
    # Elder Ray: Bull Power and Bear Power using EMA13
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Load 12h data ONCE before loop for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations (13 for Alligator jaws)
    start = 20
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(jaws[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(ema50_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long setup: bullish Alligator alignment + positive Bull Power + above 12h EMA50
            if (jaws[i] < teeth[i] and teeth[i] < lips[i] and 
                bull_power[i] > 0 and price > ema50_12h_aligned[i]):
                position = 1
                signals[i] = position_size
            # Short setup: bearish Alligator alignment + negative Bear Power + below 12h EMA50
            elif (jaws[i] > teeth[i] and teeth[i] > lips[i] and 
                  bear_power[i] < 0 and price < ema50_12h_aligned[i]):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Alligator alignment breaks (jaws > teeth OR teeth < lips)
            if jaws[i] > teeth[i] or teeth[i] < lips[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Alligator alignment breaks (jaws < teeth OR teeth > lips)
            if jaws[i] < teeth[i] or teeth[i] > lips[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_Alligator_ElderRay_12hEMA50"
timeframe = "6h"
leverage = 1.0