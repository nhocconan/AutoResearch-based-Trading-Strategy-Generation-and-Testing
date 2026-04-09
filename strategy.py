#!/usr/bin/env python3
# 1d_weekly_higher_high_lower_low_volume_v1
# Hypothesis: 1d strategy using 1w trend filter and daily higher high/lower low breakouts with volume confirmation.
# Long: Price makes higher high than previous day with volume > 1.5x 20-day average and weekly close > weekly open.
# Short: Price makes lower low than previous day with volume > 1.5x 20-day average and weekly close < weekly open.
# Exit: Opposite condition triggers (long exits on lower low, short exits on higher high).
# Uses weekly trend filter: only long when weekly close > weekly open, only short when weekly close < weekly open.
# Target: 7-25 trades/year to minimize fee drag while capturing multi-day momentum.
# Works in both bull and bear markets by following weekly trend with daily breakout confirmation.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_weekly_higher_high_lower_low_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_prices = prices['open'].values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    open_1w = df_1w['open'].values
    
    # Align 1w data to 1d
    close_1w_aligned = align_htf_to_ltf(prices, df_1w, close_1w)
    open_1w_aligned = align_htf_to_ltf(prices, df_1w, open_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(1, n):  # Start from second day to compare with previous
        # Skip if any required data is NaN
        if (np.isnan(volume_ma[i]) or np.isnan(close[i]) or
            np.isnan(high[i]) or np.isnan(low[i]) or
            np.isnan(volume[i]) or np.isnan(open_prices[i]) or
            np.isnan(close_1w_aligned[i]) or np.isnan(open_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-day average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        # Weekly bullish candle: close > open
        weekly_bullish = close_1w_aligned[i] > open_1w_aligned[i]
        # Weekly bearish candle: close < open
        weekly_bearish = close_1w_aligned[i] < open_1w_aligned[i]
        
        if position == 1:  # Long position
            # Exit: Price makes lower low than previous day
            if low[i] < low[i-1]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price makes higher high than previous day
            if high[i] > high[i-1]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long entry: Higher high with volume and weekly bullish
            if (high[i] > high[i-1] and      # Higher high
                volume_confirmed and         # Volume spike
                weekly_bullish):             # Weekly uptrend
                position = 1
                signals[i] = 0.25
            # Short entry: Lower low with volume and weekly bearish
            elif (low[i] < low[i-1] and      # Lower low
                  volume_confirmed and       # Volume spike
                  weekly_bearish):           # Weekly downtrend
                position = -1
                signals[i] = -0.25
    
    return signals