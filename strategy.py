# 12H_GoldenRatio_Trend_Pullback
# Hypothesis: Buy pullbacks in strong weekly trends using Fibonacci ratios. 
# Long when price pulls back to 0.618 Fibonacci level in uptrend + volume confirmation.
# Short when price pulls back to 0.618 level in downtrend + volume confirmation.
# Uses weekly trend filter to avoid counter-trend trades. Designed for low frequency (~15-25/year).
# Works in bull (buys dips) and bear (sells rallies) by following weekly trend.

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
    
    # Load weekly data for trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly trend using price vs 50-period EMA
    weekly_close = df_1w['close'].values
    weekly_ema50 = pd.Series(weekly_close).ewm(span=50, min_periods=50, adjust=False).mean().values
    weekly_ema50_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema50)
    
    # Calculate 12-period high/low for Fibonacci levels
    highest_12 = pd.Series(high).rolling(window=12, min_periods=12).max()
    lowest_12 = pd.Series(low).rolling(window=12, min_periods=12).min()
    range_12 = highest_12 - lowest_12
    
    # Calculate Fibonacci retracement levels (0.618 pullback level)
    fib_618_long = lowest_12 + 0.618 * range_12  # For longs in uptrend
    fib_618_short = highest_12 - 0.618 * range_12  # For shorts in downtrend
    
    # Calculate average volume for confirmation
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(12, n):
        # Skip if data not ready
        if (np.isnan(weekly_ema50_aligned[i]) or 
            np.isnan(fib_618_long[i]) or np.isnan(fib_618_short[i]) or
            np.isnan(avg_volume[i]) or volume[i] == 0 or
            np.isnan(range_12[i]) or range_12[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Get weekly trend direction
        weekly_close_val = df_1w['close'].iloc[-1] if len(df_1w) > 0 else np.nan
        weekly_ema50_val = weekly_ema50_aligned[i]
        
        if np.isnan(weekly_close_val) or np.isnan(weekly_ema50_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        weekly_trend_up = weekly_close_val > weekly_ema50_val
        weekly_trend_down = weekly_close_val < weekly_ema50_val
        
        volume_confirm = volume[i] > 1.3 * avg_volume[i]
        
        if position == 0:
            # Long: Pullback to 0.618 in uptrend + volume confirmation
            if (close[i] <= fib_618_long[i] and low[i] <= fib_618_long[i] and
                weekly_trend_up and volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: Pullback to 0.618 in downtrend + volume confirmation
            elif (close[i] >= fib_618_short[i] and high[i] >= fib_618_short[i] and
                  weekly_trend_down and volume_confirm):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: price moves against trend or breaks Fibonacci extension
            exit_signal = False
            
            if position == 1:
                # Exit long: price breaks below 0.382 level or weekly trend changes
                fib_382 = lowest_12[i] + 0.382 * range_12[i]
                if close[i] < fib_382 or not weekly_trend_up:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price breaks above 0.382 level or weekly trend changes
                fib_382 = highest_12[i] - 0.382 * range_12[i]
                if close[i] > fib_382 or not weekly_trend_down:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_GoldenRatio_Trend_Pullback"
timeframe = "12h"
leverage = 1.0