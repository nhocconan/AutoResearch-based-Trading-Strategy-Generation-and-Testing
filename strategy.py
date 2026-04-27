#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-day price action with 1-week trend filter using Donchian breakout.
# Uses weekly Donchian(20) for trend direction, daily Donchian(20) breakout with volume confirmation.
# Designed for low trade frequency (<25/year) to survive bear markets via tight entry conditions.
# Works in bull (breakouts) and bear (mean reversion at extremes) via volatility filter.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly Donchian(20) for trend direction
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    upper_1w = np.full(len(high_1w), np.nan)
    lower_1w = np.full(len(low_1w), np.nan)
    for i in range(20, len(high_1w)):
        upper_1w[i] = np.max(high_1w[i-20:i])
        lower_1w[i] = np.min(low_1w[i-20:i])
    upper_1w_aligned = align_htf_to_ltf(prices, df_1w, upper_1w)
    lower_1w_aligned = align_htf_to_ltf(prices, df_1w, lower_1w)
    
    # Get daily data for entry signals
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily Donchian(20) for breakout levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    upper_1d = np.full(len(high_1d), np.nan)
    lower_1d = np.full(len(low_1d), np.nan)
    for i in range(20, len(high_1d)):
        upper_1d[i] = np.max(high_1d[i-20:i])
        lower_1d[i] = np.min(low_1d[i-20:i])
    
    # Calculate daily 20-period volume average for confirmation
    vol_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align daily indicators to 1-minute timeframe
    upper_1d_aligned = align_htf_to_ltf(prices, df_1d, upper_1d)
    lower_1d_aligned = align_htf_to_ltf(prices, df_1d, lower_1d)
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup period
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(upper_1w_aligned[i]) or np.isnan(lower_1w_aligned[i]) or
            np.isnan(upper_1d_aligned[i]) or np.isnan(lower_1d_aligned[i]) or
            np.isnan(vol_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Weekly trend filter: price must be above/below weekly Donchian bands
        weekly_uptrend = close[i] > upper_1w_aligned[i]
        weekly_downtrend = close[i] < lower_1w_aligned[i]
        
        # Daily breakout levels
        daily_upper = upper_1d_aligned[i]
        daily_lower = lower_1d_aligned[i]
        vol_ma = vol_ma_20_1d_aligned[i]
        vol_now = volume[i]
        
        # Volume confirmation: current volume > 1.5x 20-day average
        vol_filter = vol_now > 1.5 * vol_ma
        
        # Entry logic
        if position == 0:
            # Long: daily breakout above upper band + weekly uptrend + volume
            if close[i] > daily_upper and weekly_uptrend and vol_filter:
                signals[i] = size
                position = 1
            # Short: daily breakout below lower band + weekly downtrend + volume
            elif close[i] < daily_lower and weekly_downtrend and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below daily midpoint OR volume drops below average
            midpoint = (daily_upper + daily_lower) / 2
            if close[i] < midpoint or vol_now < vol_ma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above daily midpoint OR volume drops below average
            midpoint = (daily_upper + daily_lower) / 2
            if close[i] > midpoint or vol_now < vol_ma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_WeeklyTrend_DailyDonchian_Volume"
timeframe = "1d"
leverage = 1.0