#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout + 1w EMA34 trend filter + volume confirmation
# Long when: price > Donchian upper(20) AND 1w EMA34 rising AND volume > 1.5x 20-period MA
# Short when: price < Donchian lower(20) AND 1w EMA34 falling AND volume > 1.5x 20-period MA
# Exit when: price crosses Donchian middle (20-period average of upper/lower) OR volume drops below average
# Uses Donchian for breakout structure, 1w EMA for higher-timeframe trend filter, volume for conviction
# Timeframe: 1d, HTF: 1w. Target: 30-100 total trades over 4 years (7-25/year) to avoid fee drag.
# Works in bull markets via breakouts and in bear markets via short breakdowns with trend filter.

name = "1d_Donchian20_1wEMA34_VolumeConfirm"
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
    
    # Calculate Donchian channels on 1d
    lookback = 20
    if len(high) >= lookback:
        # Rolling max/min for upper/lower bands
        upper = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
        lower = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
        middle = (upper + lower) / 2.0
    else:
        upper = np.full(n, np.nan)
        lower = np.full(n, np.nan)
        middle = np.full(n, np.nan)
    
    # Calculate EMA(34) on 1w for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    if len(close_1w) >= 34:
        ema_34_1w = pd.Series(close_1w).ewm(span=34, min_periods=34, adjust=False).mean().values
        # Rising if current > previous, falling if current < previous
        ema_rising = np.concatenate([[False], ema_34_1w[1:] > ema_34_1w[:-1]])
        ema_falling = np.concatenate([[False], ema_34_1w[1:] < ema_34_1w[:-1]])
    else:
        ema_rising = np.full(len(df_1w), False)
        ema_falling = np.full(len(df_1w), False)
    
    # Align 1w EMA trend to 1d timeframe
    ema_rising_aligned = align_htf_to_ltf(prices, df_1w, ema_rising.astype(float))
    ema_falling_aligned = align_htf_to_ltf(prices, df_1w, ema_falling.astype(float))
    
    # Volume confirmation on 1d
    vol_lookback = 20
    if len(volume) >= vol_lookback:
        vol_ma = pd.Series(volume).rolling(window=vol_lookback, min_periods=vol_lookback).mean().values
        volume_filter = volume > (1.5 * vol_ma)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(middle[i]) or 
            np.isnan(ema_rising_aligned[i]) or np.isnan(ema_falling_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: breakout above upper band + rising 1w EMA + volume filter
            if (close[i] > upper[i] and 
                ema_rising_aligned[i] == 1.0 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: breakdown below lower band + falling 1w EMA + volume filter
            elif (close[i] < lower[i] and 
                  ema_falling_aligned[i] == 1.0 and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below middle OR volume drops below average
            if (close[i] < middle[i] or not volume_filter[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above middle OR volume drops below average
            if (close[i] > middle[i] or not volume_filter[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals