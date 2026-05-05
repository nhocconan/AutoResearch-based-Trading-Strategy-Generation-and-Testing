#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 12h EMA(50) trend filter + volume spike confirmation
# Long when: price breaks above 4h Donchian(20) high AND 12h EMA(50) is rising (trend up) AND volume > 2x 20-period MA
# Short when: price breaks below 4h Donchian(20) low AND 12h EMA(50) is falling (trend down) AND volume > 2x 20-period MA
# Exit when: price returns to 4h Donchian(20) midpoint OR opposite breakout occurs
# Uses Donchian for structure, 12h EMA for HTF trend, volume for conviction
# Timeframe: 4h, HTF: 12h. Target: 75-200 total trades over 4 years (19-50/year) to avoid fee drag.

name = "4h_Donchian20_12hEMA50_Trend_VolumeConfirm"
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
    
    # Calculate volume confirmation on 4h using 20-period MA
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (2.0 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    # Calculate Donchian(20) on 4h
    if len(high) >= 20 and len(low) >= 20:
        highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
        lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
        donchian_mid = (highest_high + lowest_low) / 2.0
    else:
        highest_high = np.full(n, np.nan)
        lowest_low = np.full(n, np.nan)
        donchian_mid = np.full(n, np.nan)
    
    # Donchian breakout signals
    donchian_breakout_up = (close > highest_high) & (np.roll(close, 1) <= np.roll(highest_high, 1))
    donchian_breakout_down = (close < lowest_low) & (np.roll(close, 1) >= np.roll(lowest_low, 1))
    donchian_revert_mid = np.abs(close - donchian_mid) < 0.001 * close  # approximate midpoint return
    
    # Get 12h data ONCE before loop for EMA(50) trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:  # need sufficient data for EMA(50)
        return np.zeros(n)
    
    # Calculate EMA(50) on 12h close
    close_12h = df_12h['close'].values
    if len(close_12h) >= 50:
        ema_50 = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
        # Trend filter: EMA rising (current > previous) for long, falling for short
        ema_rising = np.roll(ema_50, 1) < ema_50  # current > previous
        ema_falling = np.roll(ema_50, 1) > ema_50  # current < previous
    else:
        ema_50 = np.full(len(close_12h), np.nan)
        ema_rising = np.full(len(close_12h), False)
        ema_falling = np.full(len(close_12h), False)
    
    # Align 12h EMA trend to 4h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    ema_rising_aligned = align_htf_to_ltf(prices, df_12h, ema_rising.astype(float))
    ema_falling_aligned = align_htf_to_ltf(prices, df_12h, ema_falling.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(ema_rising_aligned[i]) or 
            np.isnan(ema_falling_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Donchian breakout up + 12h EMA rising + volume filter
            if (donchian_breakout_up[i] and 
                ema_rising_aligned[i] == 1.0 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Donchian breakout down + 12h EMA falling + volume filter
            elif (donchian_breakout_down[i] and 
                  ema_falling_aligned[i] == 1.0 and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to Donchian midpoint OR short breakout occurs
            if (donchian_revert_mid[i] or donchian_breakout_down[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to Donchian midpoint OR long breakout occurs
            if (donchian_revert_mid[i] or donchian_breakout_up[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals