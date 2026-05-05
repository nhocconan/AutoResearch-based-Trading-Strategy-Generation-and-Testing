#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation
# Long when: price breaks above 20-day high, volume > 2.0x 20-period average, and close > 1w EMA50
# Short when: price breaks below 20-day low, volume > 2.0x 20-period average, and close < 1w EMA50
# Exit when price returns to the opposite Donchian level (mean reversion)
# Uses Donchian channels from 1d for structure, effective in both bull (breakout continuation) and bear (mean reversion via exits) markets.
# Timeframe: 1d, HTF: 1w. Target: 30-100 total trades over 4 years (7-25/year).

name = "1d_Donchian20_Breakout_1wEMA50_VolumeSpike"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_price = prices['open'].values
    
    # Calculate volume confirmation on 1d using 20-period MA
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (2.0 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    # Get 1w data ONCE before loop for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA50 trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Donchian channels from previous 20 days
    if len(high) >= 20:
        # Rolling max of high over 20 periods (excluding current)
        high_ma_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
        # Rolling min of low over 20 periods (excluding current)
        low_ma_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
        # Shift by 1 to use previous period's values
        donchian_high = np.roll(high_ma_20, 1)
        donchian_low = np.roll(low_ma_20, 1)
        # Set first value to NaN
        donchian_high[0] = np.nan
        donchian_low[0] = np.nan
    else:
        donchian_high = np.full(n, np.nan)
        donchian_low = np.full(n, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian high, volume filter, and above 1w EMA50
            if (close[i] > donchian_high[i] and 
                open_price[i] <= donchian_high[i] and  # Ensure breakout happens on this bar
                volume_filter[i] and 
                close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian low, volume filter, and below 1w EMA50
            elif (close[i] < donchian_low[i] and 
                  open_price[i] >= donchian_low[i] and  # Ensure breakdown happens on this bar
                  volume_filter[i] and 
                  close[i] < ema_50_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns below Donchian low (mean reversion)
            if close[i] < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns above Donchian high (mean reversion)
            if close[i] > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals