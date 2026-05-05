#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + 1d EMA50 trend filter + volume confirmation
# Long when: price breaks above Donchian(20) high AND 1d EMA50 is rising AND volume > 1.5x 20-period MA
# Short when: price breaks below Donchian(20) low AND 1d EMA50 is falling AND volume > 1.5x 20-period MA
# Exit when: price touches Donchian(20) midpoint OR volume drops below average
# Uses Donchian for structure, 1d EMA50 for HTF trend, volume for conviction
# Timeframe: 12h, HTF: 1d. Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.

name = "12h_Donchian20_1dEMA50_VolumeConfirm"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Donchian(20) on 12h
    if len(high) >= 20:
        donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
        donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
        donch_mid = (donch_high + donch_low) / 2.0
    else:
        donch_high = np.full(n, np.nan)
        donch_low = np.full(n, np.nan)
        donch_mid = np.full(n, np.nan)
    
    # Breakout signals
    breakout_up = close > donch_high  # price breaks above upper band
    breakout_down = close < donch_low  # price breaks below lower band
    
    # Volume confirmation on 12h
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
        volume_exit = volume < vol_ma_20  # exit when volume drops below average
    else:
        volume_filter = np.zeros(n, dtype=bool)
        volume_exit = np.ones(n, dtype=bool)  # exit if not enough data
    
    # Get 1d data ONCE before loop for EMA50 trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # need sufficient data for EMA50
        return np.zeros(n)
    
    # Calculate EMA(50) on 1d
    close_1d = df_1d['close'].values
    if len(close_1d) >= 50:
        ema_50 = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
        # EMA50 trend: rising if current > previous, falling if current < previous
        ema_50_rising = np.zeros(len(ema_50), dtype=bool)
        ema_50_falling = np.zeros(len(ema_50), dtype=bool)
        ema_50_rising[1:] = ema_50[1:] > ema_50[:-1]
        ema_50_falling[1:] = ema_50[1:] < ema_50[:-1]
    else:
        ema_50 = np.full(len(df_1d), np.nan)
        ema_50_rising = np.zeros(len(df_1d), dtype=bool)
        ema_50_falling = np.zeros(len(df_1d), dtype=bool)
    
    # Align 1d EMA50 trend to 12h timeframe
    ema_50_rising_aligned = align_htf_to_ltf(prices, df_1d, ema_50_rising.astype(float))
    ema_50_falling_aligned = align_htf_to_ltf(prices, df_1d, ema_50_falling.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(breakout_up[i]) or np.isnan(breakout_down[i]) or 
            np.isnan(donch_mid[i]) or np.isnan(volume_filter[i]) or 
            np.isnan(ema_50_rising_aligned[i]) or np.isnan(ema_50_falling_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: breakout up + EMA50 rising + volume filter
            if (breakout_up[i] and 
                ema_50_rising_aligned[i] == 1.0 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: breakout down + EMA50 falling + volume filter
            elif (breakout_down[i] and 
                  ema_50_falling_aligned[i] == 1.0 and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price touches midpoint OR volume drops
            if (close[i] <= donch_mid[i] or volume_exit[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price touches midpoint OR volume drops
            if (close[i] >= donch_mid[i] or volume_exit[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals