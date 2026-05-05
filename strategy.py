#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d volume spike and 1w EMA34 trend filter
# Long when: price breaks above upper Donchian(20), volume > 2.0x 48-period average (1d equivalent), and close > 1w EMA34
# Short when: price breaks below lower Donchian(20), volume > 2.0x 48-period average, and close < 1w EMA34
# Exit when price returns to the midpoint of the Donchian channel (mean reversion)
# Uses Donchian breakouts for structure, volume spike on 1d for conviction, and 1w EMA for major trend filter
# Timeframe: 12h, HTF: 1d/1w. Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.

name = "12h_Donchian20_Breakout_1wEMA34_1dVolumeSpike"
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
    open_price = prices['open'].values
    
    # Calculate volume confirmation on 12h using 48-period MA (equivalent to 1d lookback: 48 * 12h = 24h * 2 = 2d, adjusted for 1d)
    if len(volume) >= 48:
        vol_ma_48 = pd.Series(volume).rolling(window=48, min_periods=48).mean().values
        volume_filter = volume > (2.0 * vol_ma_48)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    # Get 1w data ONCE before loop for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA34 trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Get 1d data ONCE before loop for Donchian channel calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Donchian(20) from previous 1d bar
    if len(high_1d) >= 20:
        # Use rolling window of 20 on 1d data, then shift by 1 to use previous completed day
        high_ma_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).mean().values
        low_ma_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).mean().values
        # For Donchian, we want max(high) and min(low) over 20 periods, not mean
        # Correct approach: rolling max/min
        high_max_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
        low_min_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
        # Shift by 1 to use only completed 1d bars (no look-ahead)
        upper_channel = np.roll(high_max_20, 1)
        lower_channel = np.roll(low_min_20, 1)
        upper_channel[0] = np.nan
        lower_channel[0] = np.nan
    else:
        upper_channel = np.full(len(close_1d), np.nan)
        lower_channel = np.full(len(close_1d), np.nan)
    
    # Align Donchian levels and 1w EMA to 12h timeframe
    upper_channel_aligned = align_htf_to_ltf(prices, df_1d, upper_channel)
    lower_channel_aligned = align_htf_to_ltf(prices, df_1d, lower_channel)
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)  # already computed above
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(upper_channel_aligned[i]) or 
            np.isnan(lower_channel_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above upper Donchian, volume filter, and above 1w EMA34
            if (close[i] > upper_channel_aligned[i] and 
                open_price[i] <= upper_channel_aligned[i] and  # Ensure breakout happens on this bar
                volume_filter[i] and 
                close[i] > ema_34_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below lower Donchian, volume filter, and below 1w EMA34
            elif (close[i] < lower_channel_aligned[i] and 
                  open_price[i] >= lower_channel_aligned[i] and  # Ensure breakdown happens on this bar
                  volume_filter[i] and 
                  close[i] < ema_34_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns below midpoint of Donchian channel (mean reversion)
            midpoint = (upper_channel_aligned[i] + lower_channel_aligned[i]) / 2.0
            if close[i] < midpoint:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns above midpoint of Donchian channel (mean reversion)
            midpoint = (upper_channel_aligned[i] + lower_channel_aligned[i]) / 2.0
            if close[i] > midpoint:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals