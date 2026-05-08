#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout + weekly volume filter + monthly trend filter
# Donchian breakout captures breakouts in both bull and bear markets.
# Weekly volume filter ensures breakouts have institutional participation.
# Monthly trend filter avoids counter-trend trades in strong trends.
# Targets 10-20 trades per year (~40-80 total over 4 years) to minimize fee drag.

name = "1d_Donchian20_WeeklyVolume_MonthlyTrend"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian(20) channels
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Weekly volume filter: volume > 1.5x 4-week average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values  # ~4 weeks
    vol_filter = volume > (vol_ma * 1.5)
    
    # Get monthly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate EMA10 on weekly close for trend filter
    close_1w = df_1w['close'].values
    ema10_1w = pd.Series(close_1w).ewm(span=10, adjust=False, min_periods=10).mean().values
    ema10_1w_aligned = align_htf_to_ltf(prices, df_1w, ema10_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need enough data for Donchian
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(ema10_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        donch_high_val = donch_high[i]
        donch_low_val = donch_low[i]
        close_val = close[i]
        vol_filter_val = vol_filter[i]
        ema10_1w_val = ema10_1w_aligned[i]
        
        if position == 0:
            # Enter long: price breaks above Donchian high, volume filter, price above weekly EMA10 (uptrend)
            if close_val > donch_high_val and vol_filter_val and close_val > ema10_1w_val:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below Donchian low, volume filter, price below weekly EMA10 (downtrend)
            elif close_val < donch_low_val and vol_filter_val and close_val < ema10_1w_val:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below Donchian low or volume filter fails
            if close_val < donch_low_val or not vol_filter_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above Donchian high or volume filter fails
            if close_val > donch_high_val or not vol_filter_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals