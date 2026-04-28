#!/usr/bin/env python3
"""
6h_LR_Channel_Breakout_1wTrend_Pullback
Hypothesis: Trade pullbacks within the weekly linear regression channel on 6h timeframe.
In bull markets: buy near lower channel line during uptrend (weekly LR slope up).
In bear markets: sell near upper channel line during downtrend (weekly LR slope down).
Uses weekly LR channel (60-period) for structure and 6h RSI(2) for oversold/overbought entries.
Targets 15-25 trades/year with high win rate by trading mean reversion within the trend.
"""

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
    
    # Get weekly data for trend and channel
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    
    # Calculate weekly linear regression channel (60-period)
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Linear regression: y = mx + b over 60 periods
    def linear_regression(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan), np.full_like(arr, np.nan)
        m = np.full(len(arr), np.nan)
        b = np.full(len(arr), np.nan)
        for i in range(period-1, len(arr)):
            y = arr[i-period+1:i+1]
            x = np.arange(period)
            if np.all(~np.isnan(y)):
                # Calculate slope and intercept
                A = np.vstack([x, np.ones(len(x))]).T
                slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
                m[i] = slope
                b[i] = intercept
        return m, b
    
    # Get slope and intercept for weekly close
    slope_1w, intercept_1w = linear_regression(close_1w, 60)
    
    # Calculate weekly LR value (middle of channel)
    lr_1w = slope_1w * np.arange(len(close_1w)) + intercept_1w
    
    # Calculate weekly high/low for channel width
    # Use highest high and lowest low over 60 periods for channel width
    def rolling_extreme(arr, period, func):
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        result = np.full(len(arr), np.nan)
        for i in range(period-1, len(arr)):
            window = arr[i-period+1:i+1]
            if np.all(~np.isnan(window)):
                result[i] = func(window)
        return result
    
    highest_high_1w = rolling_extreme(high_1w, 60, np.max)
    lowest_low_1w = rolling_extreme(low_1w, 60, np.min)
    
    # Channel width based on price range
    channel_width_1w = (highest_high_1w - lowest_low_1w) * 0.5  # Half the range
    
    # Upper and lower channel lines
    upper_channel_1w = lr_1w + channel_width_1w
    lower_channel_1w = lr_1w - channel_width_1w
    
    # Align weekly data to 6h timeframe
    lr_1w_aligned = align_htf_to_ltf(prices, df_1w, lr_1w)
    upper_channel_1w_aligned = align_htf_to_ltf(prices, df_1w, upper_channel_1w)
    lower_channel_1w_aligned = align_htf_to_ltf(prices, df_1w, lower_channel_1w)
    slope_1w_aligned = align_htf_to_ltf(prices, df_1w, slope_1w)
    
    # 6h RSI(2) for oversold/overbought signals
    def rsi(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        delta = np.diff(arr, prepend=arr[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.full_like(arr, np.nan)
        avg_loss = np.full_like(arr, np.nan)
        # First average
        if len(gain) >= period:
            avg_gain[period-1] = np.mean(gain[0:period])
            avg_loss[period-1] = np.mean(loss[0:period])
        # Wilder smoothing
        for i in range(period, len(arr)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi_val = 100 - (100 / (1 + rs))
        return rsi_val
    
    rsi_2 = rsi(close, 2)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(lr_1w_aligned[i]) or 
            np.isnan(upper_channel_1w_aligned[i]) or 
            np.isnan(lower_channel_1w_aligned[i]) or
            np.isnan(slope_1w_aligned[i]) or
            np.isnan(rsi_2[i])):
            signals[i] = 0.0
            continue
        
        # Weekly trend: up if slope positive, down if slope negative
        weekly_uptrend = slope_1w_aligned[i] > 0
        weekly_downtrend = slope_1w_aligned[i] < 0
        
        # Entry conditions: pullback to channel extreme in trend direction
        # Long: price near lower channel in weekly uptrend + RSI(2) oversold
        long_entry = (weekly_uptrend and 
                     close[i] <= lower_channel_1w_aligned[i] * 1.005 and  # Within 0.5% of lower channel
                     rsi_2[i] <= 15)  # Oversold
        
        # Short: price near upper channel in weekly downtrend + RSI(2) overbought
        short_entry = (weekly_downtrend and 
                      close[i] >= upper_channel_1w_aligned[i] * 0.995 and  # Within 0.5% of upper channel
                      rsi_2[i] >= 85)  # Overbought
        
        # Exit conditions: middle of channel or trend change
        long_exit = (close[i] >= lr_1w_aligned[i] or 
                    not weekly_uptrend)
        short_exit = (close[i] <= lr_1w_aligned[i] or 
                     not weekly_downtrend)
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_LR_Channel_Breakout_1wTrend_Pullback"
timeframe = "6h"
leverage = 1.0