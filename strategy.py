#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout + volume confirmation + weekly trend filter
# Long when price breaks above 20-day high AND weekly EMA(20) rising AND volume > 1.5x average
# Short when price breaks below 20-day low AND weekly EMA(20) falling AND volume > 1.5x average
# Exit when price crosses opposite Donchian level or volume drops below threshold
# Uses weekly timeframe to reduce trade frequency, targets 30-100 total trades over 4 years
# Works in bull markets by capturing breakouts, avoids false signals in bear via trend filter

name = "1d_donchian20_weekly_ema_vol_v1"
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
    
    # Donchian channels (20-period) on daily timeframe
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Weekly EMA(20) for trend filter
    df_1w = get_htf_data(prices, '1w')
    weekly_close = df_1w['close'].values
    
    ema_20 = pd.Series(weekly_close).ewm(span=20, min_periods=20, adjust=False).mean().values
    ema_20_prev = np.roll(ema_20, 1)
    ema_20_prev[0] = ema_20[0]
    ema_rising = ema_20 > ema_20_prev
    
    # Align weekly EMA signals to daily timeframe
    ema_rising_aligned = align_htf_to_ltf(prices, df_1w, ema_rising)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 1.5 * volume_ma.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(ema_rising_aligned[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions: price crosses opposite Donchian level
        if position == 1:  # long position
            if close[i] <= lowest_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if close[i] >= highest_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + weekly trend + volume confirmation
            # Long: break above 20-day high AND weekly EMA rising AND volume confirmation
            if (close[i] > highest_high[i] and ema_rising_aligned[i] and volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below 20-day low AND weekly EMA falling AND volume confirmation
            elif (close[i] < lowest_low[i] and not ema_rising_aligned[i] and volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
    
    return signals