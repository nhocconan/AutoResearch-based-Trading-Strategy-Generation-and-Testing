# 6h_WeeklyDonchian_Volume_Filter
# Hypothesis: 6h Donchian breakout with weekly trend filter and volume confirmation.
# Long when price breaks above 6h Donchian(20) high, weekly trend is up, and volume > 1.5x average.
# Short when price breaks below 6h Donchian(20) low, weekly trend is down, and volume > 1.5x average.
# Weekly trend determined by EMA(21) on weekly closes.
# Volume filter reduces false breakouts. Weekly trend filter avoids counter-trend trades.
# Works in bull markets (follows weekly uptrend) and bear markets (follows weekly downtrend).
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.

name = "6h_WeeklyDonchian_Volume_Filter"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 25:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # 6h Donchian channels (20-period)
    highest_high = np.full_like(high, np.nan)
    lowest_low = np.full_like(low, np.nan)
    for i in range(20, n):
        highest_high[i] = np.max(high[i-19:i+1])
        lowest_low[i] = np.min(low[i-19:i+1])
    
    # Weekly EMA(21) for trend
    close_1w_series = pd.Series(close_1w)
    ema_21_1w = close_1w_series.ewm(span=21, adjust=False, min_periods=21).mean().values
    weekly_trend_up = ema_21_1w[1:] > ema_21_1w[:-1]
    weekly_trend_up = np.concatenate([[False], weekly_trend_up])
    
    # Align weekly trend to 6h
    weekly_trend_up_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend_up.astype(float))
    
    # Volume average (20-period)
    vol_avg = np.full_like(volume, np.nan)
    for i in range(20, n):
        vol_avg[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need enough data for Donchian and volume avg
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(weekly_trend_up_aligned[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price above Donchian high, weekly up, volume spike
            if (close[i] > highest_high[i] and 
                weekly_trend_up_aligned[i] and 
                volume[i] > 1.5 * vol_avg[i]):
                signals[i] = 0.25
                position = 1
            # Short breakdown: price below Donchian low, weekly down, volume spike
            elif (close[i] < lowest_low[i] and 
                  not weekly_trend_up_aligned[i] and 
                  volume[i] > 1.5 * vol_avg[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price below Donchian low or weekly trend flips down
            if close[i] < lowest_low[i] or not weekly_trend_up_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price above Donchian high or weekly trend flips up
            if close[i] > highest_high[i] or weekly_trend_up_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf