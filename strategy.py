#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation
# Long when price breaks above 4h Donchian upper channel AND 1d EMA50 rising AND volume > 1.5x 20-period average
# Short when price breaks below 4h Donchian lower channel AND 1d EMA50 falling AND volume > 1.5x 20-period average
# Exit when price crosses 4h Donchian midpoint (mean reversion) OR 1d EMA50 direction reverses
# Uses 4h primary timeframe with 1d HTF for EMA trend filter
# Donchian channels provide clear breakout zones based on recent price extremes
# EMA50 filter ensures we trade with the intermediate trend, reducing whipsaw
# Volume confirmation filters low-momentum breakouts
# Discrete sizing (0.25) to limit fee drag and manage drawdown
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe

name = "4h_Donchian20_Breakout_1dEMA50_Trend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Align EMA50 to 4h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Calculate 4h Donchian channels (20-period)
    if len(high) >= 20:
        # Upper channel: highest high of last 20 periods
        upper_channel = pd.Series(high).rolling(window=20, min_periods=20).max().values
        # Lower channel: lowest low of last 20 periods
        lower_channel = pd.Series(low).rolling(window=20, min_periods=20).min().values
        # Midpoint: average of upper and lower channels
        midpoint = (upper_channel + lower_channel) / 2
    else:
        upper_channel = np.full(n, np.nan)
        lower_channel = np.full(n, np.nan)
        midpoint = np.full(n, np.nan)
    
    # Volume confirmation: volume > 1.5x 20-period average
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_aligned[i]) or 
            np.isnan(upper_channel[i]) or 
            np.isnan(lower_channel[i]) or 
            np.isnan(midpoint[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above upper channel AND EMA50 rising AND volume spike
            if (close[i] > upper_channel[i] and 
                ema_50_aligned[i] > ema_50_aligned[i-1] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below lower channel AND EMA50 falling AND volume spike
            elif (close[i] < lower_channel[i] and 
                  ema_50_aligned[i] < ema_50_aligned[i-1] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below midpoint (mean reversion) OR EMA50 starts falling
            if close[i] < midpoint[i] or ema_50_aligned[i] < ema_50_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above midpoint (mean reversion) OR EMA50 starts rising
            if close[i] > midpoint[i] or ema_50_aligned[i] > ema_50_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals