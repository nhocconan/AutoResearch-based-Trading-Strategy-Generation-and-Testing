#!/usr/bin/env python3
"""
12h_TRIX_VolumeSpike_WeeklyTrend
Hypothesis: TRIX (15-period) momentum with volume spike confirmation and weekly trend filter captures strong momentum moves in both bull and bear markets. Weekly trend ensures alignment with higher timeframe direction, while volume spike filters low-probability signals. TRIX smooths price to reduce whipsaws, making it suitable for 12h timeframe with low trade frequency.
"""

name = "12h_TRIX_VolumeSpike_WeeklyTrend"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get weekly data for trend filter (once before loop)
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate 15-period EMA of close for TRIX (triple smoothed EMA)
    ema1 = pd.Series(close).ewm(span=15, adjust=False, min_periods=15).mean()
    ema2 = ema1.ewm(span=15, adjust=False, min_periods=15).mean()
    ema3 = ema2.ewm(span=15, adjust=False, min_periods=15).mean()
    # TRIX = percentage change of triple EMA
    trix = 100 * (ema3 - ema3.shift(1)) / ema3.shift(1)
    trix = trix.fillna(0).values  # first value is 0 due to shift
    
    # Weekly EMA34 for trend filter
    ema34_weekly = pd.Series(df_weekly['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema34_weekly)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # start after TRIX warmup
        if position == 0:
            # LONG: TRIX crosses above zero with volume spike and above weekly EMA34 (uptrend)
            if (trix[i] > 0 and trix[i-1] <= 0 and 
                volume_spike[i] and 
                close[i] > trend_weekly_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: TRIX crosses below zero with volume spike and below weekly EMA34 (downtrend)
            elif (trix[i] < 0 and trix[i-1] >= 0 and 
                  volume_spike[i] and 
                  close[i] < trend_weekly_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: TRIX crosses below zero or trend turns down
            if (trix[i] < 0 and trix[i-1] >= 0) or \
               (close[i] < trend_weekly_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: TRIX crosses above zero or trend turns up
            if (trix[i] > 0 and trix[i-1] <= 0) or \
               (close[i] > trend_weekly_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals