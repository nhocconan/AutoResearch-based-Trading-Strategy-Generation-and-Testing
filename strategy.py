#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_WilliamsVixFix_WeeklyTrend_Filter"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data once for Williams Vix Fix
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 10:
        return np.zeros(n)
    
    # Calculate Williams Vix Fix on weekly: wvf = ((highest(high, n) - low) / highest(high, n)) * 100
    # Using 22-period lookback (approx monthly)
    highest_high = pd.Series(df_w['high']).rolling(window=22, min_periods=22).max().values
    wvf = ((highest_high - df_w['low'].values) / highest_high) * 100
    wvf = np.nan_to_num(wvf, nan=0.0)
    wvf_aligned = align_htf_to_ltf(prices, df_w, wvf)
    
    # Get daily data once for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 20-period EMA for daily trend
    close_1d = df_1d['close'].values
    ema20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # warmup
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(wvf_aligned[i]) or np.isnan(ema20_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        wvf_val = wvf_aligned[i]
        ema_val = ema20_1d_aligned[i]
        
        if position == 0:
            # Enter long when Vix Fix is high (fear) and price above weekly EMA
            # High Vix Fix indicates market fear, potential reversal
            if (wvf_val > 80 and close[i] > ema_val):
                signals[i] = 0.25
                position = 1
            # Enter short when Vix Fix is low (complacency) and price below weekly EMA
            # Low Vix Fix indicates complacency, potential top
            elif (wvf_val < 20 and close[i] < ema_val):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long when fear subsides or price crosses below EMA
            if (wvf_val < 40 or close[i] < ema_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short when complacency ends or price crosses above EMA
            if (wvf_val > 60 or close[i] > ema_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Uses Williams Vix Fix (fear/greed gauge) from weekly data with daily EMA trend filter.
# - Enters long when Vix Fix > 80 (extreme fear) and price above daily EMA (contrarian entry in fear)
# - Enters short when Vix Fix < 20 (extreme complacency/greed) and price below daily EMA (fade greed)
# - Exits when Vix Fix normalizes or price crosses daily EMA
# - Williams Vix Fix measures market fear: high values = fear/panic, low values = complacency
# - Works in both bull and bear markets by fading extremes of market sentiment
# - Weekly calculation avoids noise, daily EMA provides trend context
# - Target: 50-120 total trades over 4 years (12-30/year) to minimize fee drag
# - Position size: 0.25 for balanced risk/return in volatile markets