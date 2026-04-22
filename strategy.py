#!/usr/bin/env python3
"""
Hypothesis: 6-hour Williams %R with 12-hour Trend Filter and Volume Confirmation.
Long when Williams %R < -80 (oversold) AND 12-hour EMA50 is rising AND volume > 20-period average.
Short when Williams %R > -20 (overbought) AND 12-hour EMA50 is falling AND volume > 20-period average.
Exit when Williams %R crosses -50 (mean reversion) or trend changes.
Williams %R identifies overextended moves; EMA50 filter ensures trend alignment; volume confirms institutional interest.
Works in bull markets by buying oversold dips in uptrends and in bear markets by selling overbought rallies in downtrends.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Williams %R (14-period)
    period = 14
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when high == low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Load 12-hour data for EMA50 trend filter - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_12h_rising = ema_12h > np.roll(ema_12h, 1)  # Current > previous
    ema_12h_rising[0] = False  # First value has no previous
    ema_12h_rising_aligned = align_htf_to_ltf(prices, df_12h, ema_12h_rising)
    
    ema_12h_falling = ema_12h < np.roll(ema_12h, 1)  # Current < previous
    ema_12h_falling[0] = False
    ema_12h_falling_aligned = align_htf_to_ltf(prices, df_12h, ema_12h_falling)
    
    # Volume filter: 20-period average
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(williams_r[i]) or np.isnan(avg_volume[i]) or 
            np.isnan(ema_12h_rising_aligned[i]) or np.isnan(ema_12h_falling_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Oversold + rising trend + volume confirmation
            if (williams_r[i] < -80 and ema_12h_rising_aligned[i] and 
                volume[i] > avg_volume[i]):
                signals[i] = 0.25
                position = 1
            # Short: Overbought + falling trend + volume confirmation
            elif (williams_r[i] > -20 and ema_12h_falling_aligned[i] and 
                  volume[i] > avg_volume[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Williams %R crosses above -50 (mean reversion) OR trend turns down
                if williams_r[i] > -50 or not ema_12h_rising_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Williams %R crosses below -50 (mean reversion) OR trend turns up
                if williams_r[i] < -50 or not ema_12h_falling_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_WilliamsR_12hEMA50_Volume"
timeframe = "6h"
leverage = 1.0