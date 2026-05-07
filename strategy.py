# 1d Weekly Trend with Daily Volume Filter
# Hypothesis: Weekly trend direction provides high-probability bias, with daily volume confirmation filtering false signals. Works in both bull (trend following) and bear (counter-trend reversals at extremes) by using weekly extremes as support/resistance. Target: 10-20 trades/year to minimize fee drag.

#!/usr/bin/env python3
name = "1d_Weekly_Trend_With_Daily_Volume_Filter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 2:
        return np.zeros(n)
    
    # Weekly trend: price above/below weekly open
    weekly_open = df_weekly['open'].values
    weekly_close = df_weekly['close'].values
    weekly_trend = weekly_close > weekly_open  # True for up week
    
    # Align weekly trend to daily
    weekly_trend_float = weekly_trend.astype(float)
    weekly_trend_aligned = align_htf_to_ltf(prices, df_weekly, weekly_trend_float)
    
    # Weekly high/low for support/resistance
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_high_aligned = align_htf_to_ltf(prices, df_weekly, weekly_high)
    weekly_low_aligned = align_htf_to_ltf(prices, df_weekly, weekly_low)
    
    # Daily volume filter: > 1.8x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > 1.8 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for volume MA
    
    for i in range(start_idx, n):
        if np.isnan(weekly_trend_aligned[i]) or np.isnan(weekly_high_aligned[i]) or np.isnan(weekly_low_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Weekly uptrend + price near weekly low + volume
            if (weekly_trend_aligned[i] > 0.5 and 
                close[i] <= weekly_low_aligned[i] * 1.02 and  # Within 2% of weekly low
                vol_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: Weekly downtrend + price near weekly high + volume
            elif (weekly_trend_aligned[i] < 0.5 and 
                  close[i] >= weekly_high_aligned[i] * 0.98 and  # Within 2% of weekly high
                  vol_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Weekly trend reversal or price reaches weekly high
            if weekly_trend_aligned[i] < 0.5 or close[i] >= weekly_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Weekly trend reversal or price reaches weekly low
            if weekly_trend_aligned[i] > 0.5 or close[i] <= weekly_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals