#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: Daily Range Breakout with Weekly Trend Filter (1d)
# Hypothesis: Price breaking above/below the previous day's high/low indicates
# continuation of the previous day's trend. Trend filter (price above/below
# weekly 50 EMA) ensures alignment with higher timeframe trend. Works in both
# bull and bear markets: in bull, only long breakouts; in bear, only short
# breakdowns. Volume confirmation reduces false breakouts.
# Target: 20-50 trades/year (80-200 over 4 years).

name = "1d_weekly_range_breakout_trend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 2:
        return np.zeros(n)
    
    # Get daily data for range calculation
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 2:
        return np.zeros(n)
    
    # Calculate weekly trend (50 EMA)
    weekly_close = df_weekly['close'].values
    weekly_ema_50 = pd.Series(weekly_close).ewm(span=50, min_periods=50, adjust=False).mean().values
    weekly_ema_50_aligned = align_htf_to_ltf(prices, df_weekly, weekly_ema_50)
    
    # Calculate daily data (previous day's OHLC)
    daily_high = df_daily['high'].values
    daily_low = df_daily['low'].values
    
    # Shift by 1 to use previous day's data (avoid look-ahead)
    prev_daily_high = np.roll(daily_high, 1)
    prev_daily_low = np.roll(daily_low, 1)
    # Fill first element with second to avoid using uninitialized value
    if len(prev_daily_high) > 1:
        prev_daily_high[0] = prev_daily_high[1]
        prev_daily_low[0] = prev_daily_low[1]
    else:
        prev_daily_high[0] = 0
        prev_daily_low[0] = 0
    
    # Align to daily timeframe (use previous day's levels)
    daily_high_aligned = align_htf_to_ltf(prices, df_daily, prev_daily_high)
    daily_low_aligned = align_htf_to_ltf(prices, df_daily, prev_daily_low)
    
    # Volume filter: volume > 1.3x 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.3 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(daily_high_aligned[i]) or np.isnan(daily_low_aligned[i]) or 
            np.isnan(weekly_ema_50_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price falls below previous day's low or trend turns bearish
            if low[i] < daily_low_aligned[i] or close[i] < weekly_ema_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price rises above previous day's high or trend turns bullish
            if high[i] > daily_high_aligned[i] or close[i] > weekly_ema_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Long: price breaks above previous day's high with bullish weekly trend
            if (high[i] > daily_high_aligned[i] or close[i] > daily_high_aligned[i]) and \
               close[i] > weekly_ema_50_aligned[i] and vol_filter[i]:
                position = 1
                signals[i] = 0.25
            # Short: price breaks below previous day's low with bearish weekly trend
            elif (low[i] < daily_low_aligned[i] or close[i] < daily_low_aligned[i]) and \
                 close[i] < weekly_ema_50_aligned[i] and vol_filter[i]:
                position = -1
                signals[i] = -0.25
    
    return signals