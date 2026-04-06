#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d trend filter and volume confirmation
# Long when price breaks above 20-period high AND 1d EMA(50) rising AND volume > 1.5x average
# Short when price breaks below 20-period low AND 1d EMA(50) falling AND volume > 1.5x average
# Exit when price returns to 10-period midpoint OR volume dries up
# Uses 4h timeframe targeting 75-200 trades over 4 years (19-50/year)
# Works in bull markets via breakouts and bear markets via short breakdowns
# Volume confirmation prevents false breakouts; EMA filter ensures trend alignment

name = "4h_donchian20_1d_ema_vol_v1"
timeframe = "4h"
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
    midpoint = (highest_high + lowest_low) / 2  # 10-period midpoint for exit
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * volume_ma
    
    # 1d EMA(50) for trend filter
    df_1d = get_htf_data(prices, '1d')
    daily_close = df_1d['close'].values
    ema_50 = pd.Series(daily_close).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_rising = np.diff(ema_50, prepend=ema_50[0]) > 0  # Rising if positive
    ema_50_falling = np.diff(ema_50, prepend=ema_50[0]) < 0   # Falling if negative
    
    # Align 1d EMA signals to 4h timeframe
    ema_50_rising_aligned = align_htf_to_ltf(prices, df_1d, ema_50_rising)
    ema_50_falling_aligned = align_htf_to_ltf(prices, df_1d, ema_50_falling)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or \
           np.isnan(volume_threshold[i]) or np.isnan(ema_50_rising_aligned[i]) or \
           np.isnan(ema_50_falling_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions: return to midpoint OR volume below threshold
        if position == 1:  # long position
            if close[i] <= midpoint[i] or volume[i] < volume_threshold[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if close[i] >= midpoint[i] or volume[i] < volume_threshold[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for breakouts with trend and volume confirmation
            # Bullish breakout: price above 20-period high + rising 1d EMA + volume
            if (close[i] > highest_high[i] and 
                ema_50_rising_aligned[i] and 
                volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Bearish breakout: price below 20-period low + falling 1d EMA + volume
            elif (close[i] < lowest_low[i] and 
                  ema_50_falling_aligned[i] and 
                  volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
    
    return signals