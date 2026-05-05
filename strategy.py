#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d volume spike and 1w EMA34 trend filter
# Long when: price breaks above upper Donchian(20), volume > 2.0x 2-period average (1d equivalent), and close > 1w EMA34
# Short when: price breaks below lower Donchian(20), volume > 2.0x 2-period average, and close < 1w EMA34
# Exit when price returns to opposite Donchian level (mean reversion)
# Uses Donchian structure for clear breakouts, volume spike on 1d for conviction, 1w EMA for major trend filter
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
    
    # Calculate volume confirmation on 12h using 2-period MA (equivalent to 1d lookback: 2*12h=24h≈1d)
    if len(volume) >= 2:
        vol_ma_2 = pd.Series(volume).rolling(window=2, min_periods=2).mean().values
        volume_filter = volume > (2.0 * vol_ma_2)
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
    
    # Get 1d data ONCE before loop for Donchian levels (using previous 1d bar)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Donchian(20) levels from previous 1d bar
    if len(high_1d) >= 20:
        # Use rolling window of 20 on 1d data, then shift by 1 to use previous bar
        high_series = pd.Series(high_1d)
        low_series = pd.Series(low_1d)
        donchian_high_20 = high_series.rolling(window=20, min_periods=20).max().shift(1).values
        donchian_low_20 = low_series.rolling(window=20, min_periods=20).min().shift(1).values
        # Fill NaN values from shift
        donchian_high_20[:1] = np.nan
        donchian_low_20[:1] = np.nan
    else:
        donchian_high_20 = np.full(len(high_1d), np.nan)
        donchian_low_20 = np.full(len(low_1d), np.nan)
    
    # Align Donchian levels and 1w EMA to 12h timeframe
    donchian_high_20_aligned = align_htf_to_ltf(prices, df_1d, donchian_high_20)
    donchian_low_20_aligned = align_htf_to_ltf(prices, df_1d, donchian_low_20)
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)  # already computed above
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(donchian_high_20_aligned[i]) or 
            np.isnan(donchian_low_20_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above upper Donchian, volume filter, and above 1w EMA34
            if (close[i] > donchian_high_20_aligned[i] and 
                open_price[i] <= donchian_high_20_aligned[i] and  # Ensure breakout happens on this bar
                volume_filter[i] and 
                close[i] > ema_34_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below lower Donchian, volume filter, and below 1w EMA34
            elif (close[i] < donchian_low_20_aligned[i] and 
                  open_price[i] >= donchian_low_20_aligned[i] and  # Ensure breakdown happens on this bar
                  volume_filter[i] and 
                  close[i] < ema_34_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns below lower Donchian (mean reversion to opposite level)
            if close[i] < donchian_low_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns above upper Donchian (mean reversion to opposite level)
            if close[i] > donchian_high_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals