#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout + 1w EMA(50) trend + volume confirmation
# Long when price breaks above 20-day high AND weekly close > EMA50 AND volume > 1.5x average
# Short when price breaks below 20-day low AND weekly close < EMA50 AND volume > 1.5x average
# Exit when price crosses opposite Donchian band OR weekly trend reverses
# Target: 50-150 total trades over 4 years (12-38/year)

name = "1d_donchian20_1w_ema_vol_v2"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    weekly_close = df_1w['close'].values
    
    # Weekly EMA(50)
    ema_50 = pd.Series(weekly_close).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Align weekly EMA to daily timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if required data not available
        if np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(ema_50_aligned[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions
        if position == 1:  # long position
            if close[i] <= lowest_low[i] or weekly_close[i-1] < ema_50[i-1]:  # price breaks below 20-day low OR weekly trend turns down
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if close[i] >= highest_high[i] or weekly_close[i-1] > ema_50[i-1]:  # price breaks above 20-day high OR weekly trend turns up
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + weekly trend + volume confirmation
            # Long: price breaks above 20-day high AND weekly close > EMA50 AND volume confirmation
            if (close[i] > highest_high[i-1] and weekly_close[i-1] > ema_50[i-1] and volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 20-day low AND weekly close < EMA50 AND volume confirmation
            elif (close[i] < lowest_low[i-1] and weekly_close[i-1] < ema_50[i-1] and volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
    
    return signals