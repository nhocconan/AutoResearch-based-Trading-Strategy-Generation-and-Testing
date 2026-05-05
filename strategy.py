#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 12h EMA50 trend filter + volume confirmation
# Long when: price breaks above Donchian(20) high AND 12h EMA50 rising AND volume > 1.5x 20-period MA
# Short when: price breaks below Donchian(20) low AND 12h EMA50 falling AND volume > 1.5x 20-period MA
# Exit when: price crosses opposite Donchian band OR volume drops below average
# Uses Donchian for structure, 12h EMA for trend regime, volume for conviction
# Timeframe: 4h, HTF: 12h for EMA50. Target: 75-200 total trades over 4 years (19-50/year) to avoid fee drag.

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
    else:
        donch_high = np.full(n, np.nan)
        donch_low = np.full(n, np.nan)
    
    # Breakout signals
    breakout_up = close > donch_high  # price above upper band
    breakout_down = close < donch_low  # price below lower band
    
    # Volume confirmation on 4h
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    # Get 12h data ONCE before loop for EMA50
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:  # need sufficient data for EMA50
        return np.zeros(n)
    
    # Calculate EMA(50) on 12h
    close_12h = df_12h['close'].values
    if len(close_12h) >= 50:
        ema_50 = pd.Series(close_12h).ewm(span=50, min_periods=50, adjust=False).mean().values
    else:
        ema_50 = np.full(len(df_12h), np.nan)
    
    # EMA50 trend: rising if current > previous, falling if current < previous
    ema_50_rising = np.zeros(len(ema_50), dtype=bool)
    ema_50_falling = np.zeros(len(ema_50), dtype=bool)
    for i in range(1, len(ema_50)):
        if not np.isnan(ema_50[i]) and not np.isnan(ema_50[i-1]):
            ema_50_rising[i] = ema_50[i] > ema_50[i-1]
            ema_50_falling[i] = ema_50[i] < ema_50[i-1]
    
    # Align 12h EMA50 trends to 4h timeframe
    ema_50_rising_aligned = align_htf_to_ltf(prices, df_12h, ema_50_rising.astype(float))
    ema_50_falling_aligned = align_htf_to_ltf(prices, df_12h, ema_50_falling.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(breakout_up[i]) or np.isnan(breakout_down[i]) or 
            np.isnan(volume_filter[i]) or np.isnan(ema_50_rising_aligned[i]) or 
            np.isnan(ema_50_falling_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: bullish breakout + rising trend + volume filter
            if (breakout_up[i] and 
                ema_50_rising_aligned[i] == 1.0 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: bearish breakout + falling trend + volume filter
            elif (breakout_down[i] and 
                  ema_50_falling_aligned[i] == 1.0 and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: bearish breakout OR volume drops
            if (breakout_down[i] or volume_filter[i] == False):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: bullish breakout OR volume drops
            if (breakout_up[i] or volume_filter[i] == False):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals