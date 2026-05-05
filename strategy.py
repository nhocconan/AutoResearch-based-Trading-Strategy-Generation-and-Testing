#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d volume spike and 1w EMA34 trend filter
# Long when: price breaks above 20-period Donchian high, volume > 1.8x 20-period average, and close > 1w EMA34
# Short when: price breaks below 20-period Donchian low, volume > 1.8x 20-period average, and close < 1w EMA34
# Exit when price returns to the opposite Donchian level (mean reversion) or opposite breakout
# Uses Donchian channels for structure, effective in both bull (breakout continuation) and bear (mean reversion via exits) markets.
# Timeframe: 12h, HTF: 1d/1w. Target: 50-150 total trades over 4 years (12-37/year).

name = "12h_Donchian20_Breakout_1wEMA34_VolumeSpike"
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
    
    # Calculate volume confirmation on 12h
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.8 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    # Get 1d data ONCE before loop for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1w EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate Donchian channels from previous 1d bar (20-period)
    if len(high_1d) >= 20:
        # Donchian high: highest high over past 20 days
        donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
        # Donchian low: lowest low over past 20 days
        donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    else:
        donchian_high = np.full(len(close_1d), np.nan)
        donchian_low = np.full(len(close_1d), np.nan)
    
    # Align Donchian channels to 12h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian high, volume filter, and above 1w EMA34
            if (close[i] > donchian_high_aligned[i] and 
                open_price[i] <= donchian_high_aligned[i] and  # Ensure breakout happens on this bar
                volume_filter[i] and 
                close[i] > ema_34_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian low, volume filter, and below 1w EMA34
            elif (close[i] < donchian_low_aligned[i] and 
                  open_price[i] >= donchian_low_aligned[i] and  # Ensure breakdown happens on this bar
                  volume_filter[i] and 
                  close[i] < ema_34_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns below Donchian low (mean reversion) or breaks below Donchian low (reversal)
            if close[i] < donchian_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns above Donchian high (mean reversion) or breaks above Donchian high (reversal)
            if close[i] > donchian_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals