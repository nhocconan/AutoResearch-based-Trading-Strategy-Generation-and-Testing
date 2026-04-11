#!/usr/bin/env python3
# 6h_1d_adx_breakout_v1
# Strategy: 6-hour breakout with ADX trend strength filter and 1-day directional bias
# Timeframe: 6h
# Leverage: 1.0
# Hypothesis: ADX > 25 indicates strong trend; breakouts in direction of 1-day trend have higher success.
# Works in bull markets by capturing upward breakouts with ADX confirmation.
# Works in bear markets by capturing downward breakouts when 1-day trend is down.
# Uses tight entry conditions to limit trades (~15-30/year) and avoid fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_adx_breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Daily trend: close vs open (bullish/bearish day)
    daily_bullish = df_1d['close'].values > df_1d['open'].values
    daily_bullish_aligned = align_htf_to_ltf(prices, df_1d, daily_bullish)
    
    # 6h ADX (14-period)
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    tr = np.maximum(high[1:] - low[1:], np.absolute(high[1:] - close[:-1]), np.absolute(low[1:] - close[:-1]))
    tr = np.concatenate([[np.nan], tr])  # align with original index
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            result[period-1] = np.nansum(data[:period])
            for i in range(period, len(data)):
                result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    period = 14
    tr_sum = wilder_smooth(tr, period)
    plus_dm_smooth = wilder_smooth(plus_dm, period)
    minus_dm_smooth = wilder_smooth(minus_dm, period)
    
    plus_di = 100 * plus_dm_smooth / tr_sum
    minus_di = 100 * minus_dm_smooth / tr_sum
    dx = 100 * np.absolute(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilder_smooth(dx, period)
    
    # 6h Donchian breakout (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(adx[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(daily_bullish_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # ADX trend strength filter
        strong_trend = adx[i] > 25
        
        # Breakout conditions
        breakout_up = close[i] > highest_high[i-1]  # break above prior 20-period high
        breakout_down = close[i] < lowest_low[i-1]   # break below prior 20-period low
        
        # Entry logic: breakout + strong trend + daily bias alignment
        if breakout_up and strong_trend and daily_bullish_aligned[i] and position != 1:
            position = 1
            signals[i] = 0.25
        elif breakout_down and strong_trend and not daily_bullish_aligned[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: opposite breakout
        elif position == 1 and breakout_down:
            position = 0
            signals[i] = 0.0
        elif position == -1 and breakout_up:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals