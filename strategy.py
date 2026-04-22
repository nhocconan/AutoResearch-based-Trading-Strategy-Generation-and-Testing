#!/usr/bin/env python3

"""
Hypothesis: Daily Williams %R mean reversion with weekly trend filter and volume confirmation.
Goes long when weekly trend is up and daily Williams %R crosses above oversold level (-80),
short when weekly trend is down and Williams %R crosses below overbought level (-20).
Volume confirmation reduces false signals. Designed for low trade frequency by requiring
trend alignment and momentum extremes, working in both trending and mean-reverting markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Williams %R (14-period) on daily
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when high == low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Load weekly data for trend filter - ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 10:
        return np.zeros(n)
    
    # Weekly EMA34 for trend direction
    weekly_close = df_weekly['close'].values
    ema34_weekly = pd.Series(weekly_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema34_weekly)
    
    # Volume confirmation: current volume > 1.3x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(williams_r[i]) or np.isnan(ema34_weekly_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 1.3 * vol_ma_20[i]
        
        if position == 0:
            # Long: weekly uptrend + Williams %R crosses above -80 (oversold) + volume spike
            if (ema34_weekly_aligned[i] > ema34_weekly_aligned[i-1] and 
                williams_r[i] > -80 and williams_r[i-1] <= -80 and vol_spike):
                signals[i] = 0.25
                position = 1
            # Short: weekly downtrend + Williams %R crosses below -20 (overbought) + volume spike
            elif (ema34_weekly_aligned[i] < ema34_weekly_aligned[i-1] and 
                  williams_r[i] < -20 and williams_r[i-1] >= -20 and vol_spike):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Williams %R returns to neutral range (-50) or trend changes
            exit_signal = False
            
            if position == 1:
                # Exit long: Williams %R returns above -50 or weekly trend turns down
                if williams_r[i] > -50 or ema34_weekly_aligned[i] < ema34_weekly_aligned[i-1]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Williams %R returns below -50 or weekly trend turns up
                if williams_r[i] < -50 or ema34_weekly_aligned[i] > ema34_weekly_aligned[i-1]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "Daily_WilliamsR_MeanReversion_WeeklyTrend_Volume"
timeframe = "1d"
leverage = 1.0