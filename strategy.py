#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout + 1w EMA34 trend filter + volume confirmation
# Long when: close > upper Donchian(20) AND 1w EMA34 rising AND volume > 1.5x 20-period MA
# Short when: close < lower Donchian(20) AND 1w EMA34 falling AND volume > 1.5x 20-period MA
# Exit when: price crosses middle Donchian (20-period EMA of high/low) OR volume drops below average
# Uses Donchian for breakout structure, 1w EMA for higher-timeframe trend, volume for conviction
# Timeframe: 1d, HTF: 1w. Target: 30-100 total trades over 4 years (7-25/year) to avoid fee drag.

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
    
    # Calculate Donchian(20) channels
    if len(high) >= 20:
        upper_channel = pd.Series(high).rolling(window=20, min_periods=20).max().values
        lower_channel = pd.Series(low).rolling(window=20, min_periods=20).min().values
        middle_channel = (upper_channel + lower_channel) / 2.0
    else:
        upper_channel = np.full(n, np.nan)
        lower_channel = np.full(n, np.nan)
        middle_channel = np.full(n, np.nan)
    
    # Volume confirmation on 1d
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    # Get 1w data ONCE before loop for EMA34 calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 35:  # need sufficient data for EMA34
        return np.zeros(n)
    
    # Calculate EMA(34) on 1w
    close_1w = df_1w['close'].values
    if len(close_1w) >= 34:
        ema_34 = pd.Series(close_1w).ewm(span=34, min_periods=34, adjust=False).mean().values
        # EMA rising/falling: current > previous
        ema_rising = np.zeros(len(ema_34), dtype=bool)
        ema_falling = np.zeros(len(ema_34), dtype=bool)
        ema_rising[1:] = ema_34[1:] > ema_34[:-1]
        ema_falling[1:] = ema_34[1:] < ema_34[:-1]
    else:
        ema_34 = np.full(len(df_1w), np.nan)
        ema_rising = np.zeros(len(df_1w), dtype=bool)
        ema_falling = np.zeros(len(df_1w), dtype=bool)
    
    # Align 1w EMA indicators to 1d timeframe
    ema_rising_aligned = align_htf_to_ltf(prices, df_1w, ema_rising.astype(float))
    ema_falling_aligned = align_htf_to_ltf(prices, df_1w, ema_falling.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or 
            np.isnan(middle_channel[i]) or np.isnan(volume_filter[i]) or 
            np.isnan(ema_rising_aligned[i]) or np.isnan(ema_falling_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: breakout above upper channel + rising EMA + volume filter
            if (close[i] > upper_channel[i] and 
                ema_rising_aligned[i] == 1.0 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: breakout below lower channel + falling EMA + volume filter
            elif (close[i] < lower_channel[i] and 
                  ema_falling_aligned[i] == 1.0 and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below middle channel OR volume drops below average
            if (close[i] < middle_channel[i] or not volume_filter[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above middle channel OR volume drops below average
            if (close[i] > middle_channel[i] or not volume_filter[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals