#!/usr/bin/env python3
"""
1d_WPivot_R1_S1_Breakout_Volume_TrendFilter
Strategy: Daily weekly pivot R1/S1 breakout with volume confirmation and weekly EMA50 trend filter.
Long: Price breaks above R1 + volume > 1.5x 20-period avg + price > weekly EMA50
Short: Price breaks below S1 + volume > 1.5x 20-period avg + price < weekly EMA50
Exit: Opposite pivot level touch or trend reversal
Position size: 0.25
Designed to capture breakouts in trending markets while avoiding false signals in ranging conditions.
Timeframe: 1d
"""

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
    
    # Calculate weekly pivot levels for current week using previous week's OHLC
    def calculate_pivot(high, low, close):
        # Pivot point
        pp = (high + low + close) / 3
        # Range
        range_ = high - low
        # R1 and S1 levels
        r1 = pp + range_ * 1.1 / 12
        s1 = pp - range_ * 1.1 / 12
        return r1, s1
    
    # Get weekly OHLC data
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 2:
        return np.zeros(n)
    
    # Calculate pivot levels for each week using previous week's OHLC
    weekly_r1 = np.full(len(df_weekly), np.nan)
    weekly_s1 = np.full(len(df_weekly), np.nan)
    
    for i in range(1, len(df_weekly)):
        prev_high = df_weekly.iloc[i-1]['high']
        prev_low = df_weekly.iloc[i-1]['low']
        prev_close = df_weekly.iloc[i-1]['close']
        r1_val, s1_val = calculate_pivot(prev_high, prev_low, prev_close)
        weekly_r1[i] = r1_val
        weekly_s1[i] = s1_val
    
    # Align weekly pivot levels to daily timeframe
    r1_daily = align_htf_to_ltf(prices, df_weekly, weekly_r1)
    s1_daily = align_htf_to_ltf(prices, df_weekly, weekly_s1)
    
    # Calculate weekly EMA50 for trend filter
    ema_50_weekly = pd.Series(df_weekly['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_daily = align_htf_to_ltf(prices, df_weekly, ema_50_weekly)
    
    # Calculate daily volume average (20-period)
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # warmup for indicators
        # Skip if any required data is not available
        if (np.isnan(r1_daily[i]) or np.isnan(s1_daily[i]) or 
            np.isnan(ema_50_daily[i]) or np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period average
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        # Trend filter: price above/below weekly EMA50
        trend_up = close[i] > ema_50_daily[i]
        trend_down = close[i] < ema_50_daily[i]
        
        # Breakout conditions
        breakout_up = close[i] > r1_daily[i]
        breakout_down = close[i] < s1_daily[i]
        
        # Entry signals
        if position == 0:
            # Long: breakout above R1 + volume filter + trend up
            if breakout_up and volume_filter and trend_up:
                signals[i] = 0.25
                position = 1
            # Short: breakout below S1 + volume filter + trend down
            elif breakout_down and volume_filter and trend_down:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price touches S1 or trend down
            if close[i] <= s1_daily[i] or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price touches R1 or trend up
            if close[i] >= r1_daily[i] or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WPivot_R1_S1_Breakout_Volume_TrendFilter"
timeframe = "1d"
leverage = 1.0