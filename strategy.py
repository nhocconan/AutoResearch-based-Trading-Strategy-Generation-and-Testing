#!/usr/bin/env python3
"""
1d_1W_Camarilla_Pivot_Squeeze_Reverse_v1
Hypothesis: In low-volatility squeeze (weekly Bollinger Band Width < 20th percentile), 
price often reverts to the weekly mean. We use the weekly midline (average of weekly high/low) 
as a mean-reversion target, entering when daily price touches the weekly Bollinger Bands 
(upper/lower) with rejection (close back inside bands). Weekly trend filter (price > weekly EMA20 for long, 
< for short) ensures we trade with the weekly bias. This captures mean reversion in ranging markets 
while avoiding strong trends. Works in both bull and bear as it exploits mean reversion in low volatility.
"""

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
    
    # Get weekly data for mean reversion target and trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly calculations
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly Bollinger Bands (20, 2)
    close_1w_series = pd.Series(close_1w)
    weekly_ma = close_1w_series.rolling(window=20, min_periods=20).mean().values
    weekly_std = close_1w_series.rolling(window=20, min_periods=20).std().values
    weekly_upper = weekly_ma + 2 * weekly_std
    weekly_lower = weekly_ma - 2 * weekly_std
    weekly_width = weekly_upper - weekly_lower
    
    # Weekly EMA20 for trend filter
    weekly_ema20 = close_1w_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Weekly midline (mean of high and low) as additional target
    weekly_midline = (high_1w + low_1w) / 2
    
    # Align weekly data to daily
    weekly_upper_aligned = align_htf_to_ltf(prices, df_1w, weekly_upper)
    weekly_lower_aligned = align_htf_to_ltf(prices, df_1w, weekly_lower)
    weekly_ema20_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema20)
    weekly_midline_aligned = align_htf_to_ltf(prices, df_1w, weekly_midline)
    
    # Daily Bollinger Band Width percentile (20-period) for squeeze detection
    close_series = pd.Series(close)
    daily_ma = close_series.rolling(window=20, min_periods=20).mean().values
    daily_std = close_series.rolling(window=20, min_periods=20).std().values
    daily_upper = daily_ma + 2 * daily_std
    daily_lower = daily_ma - 2 * daily_std
    daily_width = daily_upper - daily_lower
    
    # Percentile of daily BB width (lookback 50 days)
    width_percentile = np.zeros_like(daily_width)
    for i in range(50, len(daily_width)):
        window = daily_width[i-50:i]
        width_percentile[i] = (np.sum(window < daily_width[i]) / 50) * 100
    
    # Squeeze condition: weekly BB width < 20th percentile (low volatility)
    squeeze_condition = width_percentile < 20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # need enough for calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(weekly_upper_aligned[i]) or np.isnan(weekly_lower_aligned[i]) or 
            np.isnan(weekly_ema20_aligned[i]) or np.isnan(weekly_midline_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Only trade in squeeze (low volatility) conditions
        in_squeeze = squeeze_condition[i]
        
        if position == 0:
            # Long: price touches or goes below weekly lower band AND closes back inside (rejection)
            # AND price is above weekly EMA20 (weekly uptrend bias)
            if (low[i] <= weekly_lower_aligned[i] and close[i] > weekly_lower_aligned[i] and
                close[i] > weekly_ema20_aligned[i] and in_squeeze):
                signals[i] = 0.25
                position = 1
            # Short: price touches or goes above weekly upper band AND closes back inside (rejection)
            # AND price is below weekly EMA20 (weekly downtrend bias)
            elif (high[i] >= weekly_upper_aligned[i] and close[i] < weekly_upper_aligned[i] and
                  close[i] < weekly_ema20_aligned[i] and in_squeeze):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price reaches weekly midline or weekly upper band or squeeze ends
            if (close[i] >= weekly_midline_aligned[i] or 
                close[i] >= weekly_upper_aligned[i] or 
                not in_squeeze):
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price reaches weekly midline or weekly lower band or squeeze ends
            if (close[i] <= weekly_midline_aligned[i] or 
                close[i] <= weekly_lower_aligned[i] or 
                not in_squeeze):
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_1W_Camarilla_Pivot_Squeeze_Reverse_v1"
timeframe = "1d"
leverage = 1.0