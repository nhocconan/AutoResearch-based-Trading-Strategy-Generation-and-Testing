#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly pivot direction filter and volume confirmation.
# Uses Donchian channel breakouts filtered by weekly pivot bias (above/below weekly pivot) and volume > 1.5x 20-period median.
# Weekly pivot acts as regime filter: long only when price > weekly pivot, short only when price < weekly pivot.
# Works in bull (buy breakouts with bullish weekly bias) and bear (sell breakdowns with bearish weekly bias).
# Discrete position sizing (0.25) to minimize fee churn. Target: 50-150 total trades over 4 years.

name = "6h_Donchian20_WeeklyPivot_VolumeConfirm_v1"
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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate weekly pivot from prior week OHLC (using 1d data)
    # Weekly pivot = (PriorWeekHigh + PriorWeekLow + PriorWeekClose) / 3
    # We approximate weekly OHLC by rolling 5-day (assuming 5 trading days per week)
    week_high = df_1d['high'].rolling(window=5, min_periods=5).max().shift(1).values  # prior week high
    week_low = df_1d['low'].rolling(window=5, min_periods=5).min().shift(1).values    # prior week low
    week_close = df_1d['close'].rolling(window=5, min_periods=5).last().shift(1).values  # prior week close
    weekly_pivot = (week_high + week_low + week_close) / 3.0
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    
    # Calculate Donchian(20) channels
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 20-period volume median for volume confirmation
    vol_median_20 = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for Donchian and volume median
    start_idx = 20
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or
            np.isnan(vol_median_20[i]) or
            np.isnan(weekly_pivot_aligned[i])):
            signals[i] = 0.0
            if position != 0:
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Weekly pivot bias filter: long bias if price > weekly pivot, short bias if price < weekly pivot
        bullish_bias = curr_close > weekly_pivot_aligned[i]
        bearish_bias = curr_close < weekly_pivot_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 20-period volume median
        if vol_median_20[i] <= 0 or np.isnan(vol_median_20[i]):
            volume_confirm = False
        else:
            volume_confirm = curr_volume > (vol_median_20[i] * 1.5)
        
        # Donchian breakout conditions
        breakout_up = curr_high > donchian_high[i]   # break above upper channel
        breakout_down = curr_low < donchian_low[i]   # break below lower channel
        
        if position == 0:  # Flat - look for new entries
            # Long: Breakout up AND bullish bias AND volume confirmation
            if breakout_up and bullish_bias and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Breakout down AND bearish bias AND volume confirmation
            elif breakout_down and bearish_bias and volume_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on Donchian breakdown (reversal signal)
            if curr_low < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on Donchian breakout (reversal signal)
            if curr_high > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals