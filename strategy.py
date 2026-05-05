#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian Channel Breakout with 1w EMA Trend Filter
# Long when: Price breaks above 20-day Donchian high AND weekly EMA(34) is rising (bullish weekly trend)
# Short when: Price breaks below 20-day Donchian low AND weekly EMA(34) is falling (bearish weekly trend)
# Exit when price returns to the 20-day Donchian middle (mean reversion)
# Donchian breakout captures sustained momentum after consolidation
# Weekly EMA filter ensures we trade in the direction of the higher timeframe trend
# Works in both bull and bear markets by aligning with weekly trend
# Target: 30-100 total trades over 4 years (7-25/year) with discrete sizing 0.25

name = "1d_DonchianBreakout_WeeklyEMATrend"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1w data ONCE before loop for weekly EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:  # Need enough for EMA(34)
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA(34)
    ema_34_1w = pd.Series(close_1w).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate 20-day Donchian channels
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    middle_20 = (highest_20 + lowest_20) / 2
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(highest_20[i]) or 
            np.isnan(lowest_20[i]) or np.isnan(middle_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Weekly EMA trend: rising if current > previous, falling if current < previous
        if i > 100:  # Need previous value for comparison
            ema_rising = ema_34_1w_aligned[i] > ema_34_1w_aligned[i-1]
            ema_falling = ema_34_1w_aligned[i] < ema_34_1w_aligned[i-1]
        else:
            ema_rising = False
            ema_falling = False
        
        if position == 0:
            # Long: Break above Donchian high with rising weekly EMA
            if close[i] > highest_20[i] and ema_rising:
                signals[i] = 0.25
                position = 1
            # Short: Break below Donchian low with falling weekly EMA
            elif close[i] < lowest_20[i] and ema_falling:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: return to Donchian middle (mean reversion)
            if close[i] < middle_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: return to Donchian middle (mean reversion)
            if close[i] > middle_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals