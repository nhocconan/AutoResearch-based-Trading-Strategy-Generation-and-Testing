#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA21 trend filter and volume confirmation
# Donchian breakouts capture momentum in both bull and bear markets. The 1w EMA21
# provides a robust trend filter to avoid whipsaws, while volume confirmation ensures
# breakout validity. Designed for 1d timeframe to target 30-100 total trades over 4 years
# (7-25/year) with discrete sizing (0.30). Works in bull markets by buying upside
# breakouts in uptrends and in bear markets by selling downside breakouts in downtrends.

name = "1d_Donchian20_1wEMA21_Trend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA21 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)
    
    # Calculate 1w EMA21 for trend filter
    close_1w = df_1w['close'].values
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Calculate Donchian channels (20-period) on 1d data
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: 1.5x 20-period EMA on 1d volume
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start from 100 to have valid indicators
        # Skip if any value is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema_21_1w_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.5 x 20-period EMA
        volume_confirmed = volume[i] > (1.5 * vol_ema_20[i])
        
        if position == 0:
            # Long: price breaks above Donchian upper + volume confirmation + price above 1w EMA21 (uptrend)
            if (close[i] > highest_high[i] and volume_confirmed and 
                close[i] > ema_21_1w_aligned[i]):
                signals[i] = 0.30
                position = 1
            # Short: price breaks below Donchian lower + volume confirmation + price below 1w EMA21 (downtrend)
            elif (close[i] < lowest_low[i] and volume_confirmed and 
                  close[i] < ema_21_1w_aligned[i]):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: price breaks below Donchian lower OR price below 1w EMA21 (trend change)
            if close[i] < lowest_low[i] or close[i] < ema_21_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: price breaks above Donchian upper OR price above 1w EMA21 (trend change)
            if close[i] > highest_high[i] or close[i] > ema_21_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals