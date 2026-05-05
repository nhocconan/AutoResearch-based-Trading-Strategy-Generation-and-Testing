#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + 1d EMA34 trend filter + volume spike confirmation
# Long when: price breaks above 20-period Donchian high AND price > 1d EMA34 (uptrend) AND volume > 1.5x 20-period MA
# Short when: price breaks below 20-period Donchian low AND price < 1d EMA34 (downtrend) AND volume > 1.5x 20-period MA
# Exit when: price returns to 10-period Donchian midpoint OR trend reverses
# Uses Donchian channels for structure, 1d EMA for higher timeframe trend, volume for conviction
# Timeframe: 12h, HTF: 1d. Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.

name = "12h_Donchian20_Breakout_1dEMA34_VolumeConfirm"
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
    
    # Calculate volume confirmation on 12h using 20-period MA
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    # Calculate Donchian channels on 12h (20-period)
    if len(high) >= 20 and len(low) >= 20:
        donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
        donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
        donchian_mid = (donchian_high + donchian_low) / 2.0
    else:
        donchian_high = np.full(n, np.nan)
        donchian_low = np.full(n, np.nan)
        donchian_mid = np.full(n, np.nan)
    
    # Donchian breakout signals
    donchian_breakout_up = close > donchian_high
    donchian_breakout_down = close < donchian_low
    donchian_reentry = (close > donchian_mid) & (close < donchian_high)  # For long exit
    donchian_reentry_short = (close < donchian_mid) & (close > donchian_low)  # For short exit
    
    # Get 1d data ONCE before loop for EMA calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 34-period EMA on 1d timeframe
    if len(close_1d) >= 34:
        ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
        ema_rising = np.diff(ema_34_1d, prepend=np.nan) > 0
        ema_falling = np.diff(ema_34_1d, prepend=np.nan) < 0
    else:
        ema_rising = np.full(len(close_1d), False)
        ema_falling = np.full(len(close_1d), False)
    
    # Align 1d EMA trend to 12h timeframe
    ema_rising_aligned = align_htf_to_ltf(prices, df_1d, ema_rising.astype(float))
    ema_falling_aligned = align_htf_to_ltf(prices, df_1d, ema_falling.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(donchian_mid[i]) or 
            np.isnan(ema_rising_aligned[i]) or np.isnan(ema_falling_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Donchian breakout up + 1d EMA rising + volume filter
            if (donchian_breakout_up[i] and 
                ema_rising_aligned[i] == 1.0 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Donchian breakout down + 1d EMA falling + volume filter
            elif (donchian_breakout_down[i] and 
                  ema_falling_aligned[i] == 1.0 and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to Donchian midpoint OR 1d EMA turns falling
            if (donchian_reentry[i] or ema_falling_aligned[i] == 1.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to Donchian midpoint OR 1d EMA turns rising
            if (donchian_reentry_short[i] or ema_rising_aligned[i] == 1.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals