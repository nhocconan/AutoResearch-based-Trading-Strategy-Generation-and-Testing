#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 12h EMA50 trend filter + volume confirmation
# Long when: price breaks above Donchian(20) high AND 12h EMA50 trending up AND volume > 1.5x 20-period MA
# Short when: price breaks below Donchian(20) low AND 12h EMA50 trending down AND volume > 1.5x 20-period MA
# Exit when: price crosses Donchian(20) midpoint OR trend reverses
# Uses Donchian for structure, 12h EMA for regime, volume for conviction
# Timeframe: 4h, HTF: 12h for EMA. Target: 75-200 total trades over 4 years (19-50/year) to avoid fee drag.

name = "4h_Donchian20_12hEMA50_VolumeConfirm"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian(20) on 4h
    if len(high) >= 20:
        donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
        donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
        donch_mid = (donch_high + donch_low) / 2
    else:
        donch_high = np.full(n, np.nan)
        donch_low = np.full(n, np.nan)
        donch_mid = np.full(n, np.nan)
    
    # Breakout signals
    breakout_up = close > donch_high  # price above prior 20-period high
    breakout_down = close < donch_low  # price below prior 20-period low
    
    # Volume confirmation on 4h
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    # Get 12h data ONCE before loop for EMA50
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:  # need sufficient data for EMA
        return np.zeros(n)
    
    # Calculate EMA(50) on 12h
    close_12h = df_12h['close'].values
    if len(close_12h) >= 50:
        ema_50 = pd.Series(close_12h).ewm(span=50, min_periods=50, adjust=False).mean().values
    else:
        ema_50 = np.full(len(df_12h), np.nan)
    
    # EMA trend: rising if current > previous, falling if current < previous
    ema_trend_up = np.zeros(len(ema_50), dtype=bool)
    ema_trend_down = np.zeros(len(ema_50), dtype=bool)
    ema_trend_up[1:] = ema_50[1:] > ema_50[:-1]
    ema_trend_down[1:] = ema_50[1:] < ema_50[:-1]
    
    # Align 12h EMA trends to 4h timeframe
    ema_trend_up_aligned = align_htf_to_ltf(prices, df_12h, ema_trend_up.astype(float))
    ema_trend_down_aligned = align_htf_to_ltf(prices, df_12h, ema_trend_down.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(breakout_up[i]) or np.isnan(breakout_down[i]) or 
            np.isnan(volume_filter[i]) or np.isnan(ema_trend_up_aligned[i]) or 
            np.isnan(ema_trend_down_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: breakout up + uptrend + volume filter
            if (breakout_up[i] and 
                ema_trend_up_aligned[i] == 1.0 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: breakout down + downtrend + volume filter
            elif (breakout_down[i] and 
                  ema_trend_down_aligned[i] == 1.0 and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses midpoint OR trend reverses to down
            if (close[i] < donch_mid[i] or ema_trend_down_aligned[i] == 1.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses midpoint OR trend reverses to up
            if (close[i] > donch_mid[i] or ema_trend_up_aligned[i] == 1.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals