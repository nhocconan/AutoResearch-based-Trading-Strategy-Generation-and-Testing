#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 12h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation
# Long when price breaks above 12h Donchian upper channel AND 12h EMA50 > EMA200 AND volume > 1.5 * avg_volume(20)
# Short when price breaks below 12h Donchian lower channel AND 12h EMA50 < EMA200 AND volume > 1.5 * avg_volume(20)
# Exit when price touches 12h Donchian middle channel (mean of upper/lower) or opposite channel level
# Uses discrete sizing 0.25 to balance return and drawdown control
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe
# 12h Donchian provides clear structural support/resistance levels
# 12h EMA50/EMA200 filter ensures alignment with intermediate trend, reducing counter-trend trades
# Volume confirmation (1.5x) filters weak breakouts while avoiding excessive filtering
# Works in bull (trend continuation breakouts above upper channel) and bear (trend continuation breakdowns below lower channel)

name = "4h_12hDonchian20_12hEMA50Trend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:  # Need sufficient data for EMA200
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data ONCE before loop for Donchian channels and EMA trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 200:  # Need sufficient data for EMA200
        return np.zeros(n)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h Donchian(20) channels
    highest_20 = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    upper_channel = highest_20
    lower_channel = lowest_20
    middle_channel = (upper_channel + lower_channel) / 2.0
    
    # Align 12h Donchian channels to 4h timeframe (wait for completed 12h bar)
    upper_aligned = align_htf_to_ltf(prices, df_12h, upper_channel)
    lower_aligned = align_htf_to_ltf(prices, df_12h, lower_channel)
    middle_aligned = align_htf_to_ltf(prices, df_12h, middle_channel)
    
    # Calculate 12h EMA50 and EMA200 for trend filter
    close_series_12h = pd.Series(close_12h)
    ema_50_12h = close_series_12h.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200_12h = close_series_12h.ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 12h EMA values to 4h timeframe (wait for completed 12h bar)
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    ema_200_aligned = align_htf_to_ltf(prices, df_12h, ema_200_12h)
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or np.isnan(middle_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(ema_200_aligned[i]) or np.isnan(avg_volume_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 12h Donchian upper channel with 12h EMA50 > EMA200 and volume confirmation
            if (close[i] > upper_aligned[i] and close[i-1] <= upper_aligned[i-1] and 
                ema_50_aligned[i] > ema_200_aligned[i] and volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 12h Donchian lower channel with 12h EMA50 < EMA200 and volume confirmation
            elif (close[i] < lower_aligned[i] and close[i-1] >= lower_aligned[i-1] and 
                  ema_50_aligned[i] < ema_200_aligned[i] and volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price touches 12h Donchian middle channel or lower channel (profit take or reversal)
            if close[i] <= middle_aligned[i] or close[i] <= lower_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price touches 12h Donchian middle channel or upper channel (profit take or reversal)
            if close[i] >= middle_aligned[i] or close[i] >= upper_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals