# The strategy is designed for the 1-day timeframe, using weekly high/low as a trend filter and daily price action for entries.
# It aims to capture trend-following moves by going long when price breaks above the weekly high with volume confirmation,
# and short when price breaks below the weekly low with volume confirmation, only in the direction of the weekly trend.
# The weekly trend is determined by whether the current weekly close is above or below the weekly open.
# This approach should work in both bull and bear markets by following the higher timeframe trend.
# Position size is set to 0.25 to balance risk and return, and to minimize trade frequency and fee drag.

#!/usr/bin/env python3

name = "1D_WEEKLY_HIGH_LOW_BREAKOUT_VOLUME"
timeframe = "1d"
leverage = 1.0

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
    
    # Get weekly data for trend filter and breakout levels
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 2:
        return np.zeros(n)
    
    # Calculate weekly trend: 1 if weekly close > weekly open (uptrend), -1 if weekly close < weekly open (downtrend)
    weekly_open = df_weekly['open'].values
    weekly_close = df_weekly['close'].values
    weekly_trend = np.where(weekly_close > weekly_open, 1, -1)
    
    # Weekly high and low for breakout levels
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    
    # Align weekly data to daily timeframe
    weekly_trend_aligned = align_htf_to_ltf(prices, df_weekly, weekly_trend)
    weekly_high_aligned = align_htf_to_ltf(prices, df_weekly, weekly_high)
    weekly_low_aligned = align_htf_to_ltf(prices, df_weekly, weekly_low)
    
    # Volume spike: current daily volume > 2.0 x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=1).mean().values
    volume_spike = volume > 2.0 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if any critical data is not ready
        if (np.isnan(weekly_trend_aligned[i]) or np.isnan(weekly_high_aligned[i]) or 
            np.isnan(weekly_low_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Break above weekly high with volume spike in weekly uptrend
            if (high[i] > weekly_high_aligned[i] and 
                volume_spike[i] and 
                weekly_trend_aligned[i] == 1):
                signals[i] = 0.25
                position = 1
            # SHORT: Break below weekly low with volume spike in weekly downtrend
            elif (low[i] < weekly_low_aligned[i] and 
                  volume_spike[i] and 
                  weekly_trend_aligned[i] == -1):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price falls back below weekly high or weekly trend turns down
            if (close[i] < weekly_high_aligned[i] or 
                weekly_trend_aligned[i] == -1):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price rises back above weekly low or weekly trend turns up
            if (close[i] > weekly_low_aligned[i] or 
                weekly_trend_aligned[i] == 1):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals