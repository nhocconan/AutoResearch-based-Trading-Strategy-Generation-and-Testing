#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray with 1-week Trend Filter
# Elder Ray measures bull/bear power: Bull Power = High - EMA(13), Bear Power = Low - EMA(13)
# We use 1-week EMA(13) as trend filter: only take long signals when weekly trend is up (EMA rising),
# and short signals when weekly trend is down (EMA falling). This avoids counter-trend trades.
# Works in bull markets (catches strength in uptrend) and bear markets (catches weakness in downtrend).
# Target: 50-150 total trades over 4 years.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Load 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Calculate EMA(13) on weekly close
    ema_13_1w = pd.Series(close_1w).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Determine weekly trend: rising EMA = uptrend, falling EMA = downtrend
    ema_13_slope = np.diff(ema_13_1w, prepend=ema_13_1w[0])
    weekly_uptrend = ema_13_slope > 0
    weekly_downtrend = ema_13_slope < 0
    
    # Align weekly trend to 6h timeframe
    weekly_uptrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_uptrend.astype(float))
    weekly_downtrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_downtrend.astype(float))
    
    # Calculate EMA(13) on 6h close for Elder Ray
    ema_13_6h = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema_13_6h  # Strength of bulls: high above EMA
    bear_power = low - ema_13_6h   # Strength of bears: low below EMA (negative values)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(13, n):  # Start after EMA warmup
        # Skip if trend data not available
        if np.isnan(weekly_uptrend_aligned[i]) or np.isnan(weekly_downtrend_aligned[i]):
            continue
        
        # Long entry: bull power positive AND weekly uptrend
        if (bull_power[i] > 0 and 
            weekly_uptrend_aligned[i] > 0.5 and  # True when aligned
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: bear power negative AND weekly downtrend
        elif (bear_power[i] < 0 and 
              weekly_downtrend_aligned[i] > 0.5 and  # True when aligned
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: opposite Elder Ray signal or trend change
        elif position == 1 and (bull_power[i] <= 0 or weekly_uptrend_aligned[i] <= 0.5):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (bear_power[i] >= 0 or weekly_downtrend_aligned[i] <= 0.5):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "6h_ElderRay_1wTrend"
timeframe = "6h"
leverage = 1.0