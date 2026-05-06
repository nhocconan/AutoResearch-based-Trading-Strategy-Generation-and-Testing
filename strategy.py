#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation
# Long when price breaks above 20-period Donchian high AND 12h EMA50 is rising AND volume > 1.5 * avg_volume(20)
# Short when price breaks below 20-period Donchian low AND 12h EMA50 is falling AND volume > 1.5 * avg_volume(20)
# Exit when price crosses the 20-period Donchian midpoint
# Uses discrete sizing 0.25 to balance return and drawdown control
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe
# Donchian channels provide clear structure, EMA50 filters for trend direction, volume confirms participation
# Works in bull markets (continuation breakouts) and bear markets (continuation breakdowns)

name = "4h_Donchian20_12hEMA50_Trend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data ONCE before loop for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:  # Need sufficient data for EMA50
        return np.zeros(n)
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA50 for trend direction
    ema_50_12h = pd.Series(close_12h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    ema_50_12h_rising = ema_50_12h_aligned > np.roll(ema_50_12h_aligned, 1)
    ema_50_12h_falling = ema_50_12h_aligned < np.roll(ema_50_12h_aligned, 1)
    
    # Calculate Donchian(20) channels on 4h data
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    donchian_mid = (highest_high + lowest_low) / 2.0
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(ema_50_12h_aligned[i]) or
            np.isnan(avg_volume_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high with rising 12h EMA50 and volume spike
            if (close[i] > highest_high[i] and close[i-1] <= highest_high[i-1] and 
                ema_50_12h_rising[i] and volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low with falling 12h EMA50 and volume spike
            elif (close[i] < lowest_low[i] and close[i-1] >= lowest_low[i-1] and 
                  ema_50_12h_falling[i] and volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below Donchian midpoint
            if close[i] < donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above Donchian midpoint
            if close[i] > donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals