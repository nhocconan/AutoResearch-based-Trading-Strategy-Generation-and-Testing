#!/usr/bin/env python3
name = "1d_TripleEMA_WeeklyTrend_Filter"
timeframe = "1d"
leverage = 1.0

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
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate 3 EMAs on weekly
    ema8_1w = pd.Series(close_1w).ewm(span=8, adjust=False, min_periods=8).mean().values
    ema21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema55_1w = pd.Series(close_1w).ewm(span=55, adjust=False, min_periods=55).mean().values
    
    # Align weekly EMAs to daily
    ema8_1w_aligned = align_htf_to_ltf(prices, df_1w, ema8_1w)
    ema21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema21_1w)
    ema55_1w_aligned = align_htf_to_ltf(prices, df_1w, ema55_1w)
    
    # Calculate daily EMAs for entry signals
    ema8_daily = pd.Series(close).ewm(span=8, adjust=False, min_periods=8).mean().values
    ema21_daily = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema55_daily = pd.Series(close).ewm(span=55, adjust=False, min_periods=55).mean().values
    
    # Volume filter: volume > 1.5x 20-day average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma20 * 1.5)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 55  # Ensure EMAs are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema8_1w_aligned[i]) or np.isnan(ema21_1w_aligned[i]) or 
            np.isnan(ema55_1w_aligned[i]) or np.isnan(ema8_daily[i]) or 
            np.isnan(ema21_daily[i]) or np.isnan(ema55_daily[i]) or 
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: weekly EMA8 > EMA21 > EMA55 AND daily EMA8 > EMA21 > EMA55 AND volume filter
            if (ema8_1w_aligned[i] > ema21_1w_aligned[i] > ema55_1w_aligned[i] and
                ema8_daily[i] > ema21_daily[i] > ema55_daily[i] and
                vol_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: weekly EMA8 < EMA21 < EMA55 AND daily EMA8 < EMA21 < EMA55 AND volume filter
            elif (ema8_1w_aligned[i] < ema21_1w_aligned[i] < ema55_1w_aligned[i] and
                  ema8_daily[i] < ema21_daily[i] < ema55_daily[i] and
                  vol_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: weekly trend turns bearish OR daily EMA cross down
            if (ema8_1w_aligned[i] < ema21_1w_aligned[i] or
                ema8_daily[i] < ema21_daily[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: weekly trend turns bullish OR daily EMA cross up
            if (ema8_1w_aligned[i] > ema21_1w_aligned[i] or
                ema8_daily[i] > ema21_daily[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals