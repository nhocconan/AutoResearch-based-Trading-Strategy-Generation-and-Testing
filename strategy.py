#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1w trend filter and volume confirmation
# Long when: price breaks above 6h Donchian upper (20-bar high), weekly EMA50 uptrend (close > EMA50), and volume > 1.8x 24-period MA (12h equivalent)
# Short when: price breaks below 6h Donchian lower (20-bar low), weekly EMA50 downtrend (close < EMA50), and volume > 1.8x 24-period MA
# Exit: time-based exit after 3 bars (18h) to avoid whipsaw in ranging markets
# Uses weekly trend for structure (works in bull/bear via EMA50 filter), Donchian for breakouts, volume for confirmation.
# Timeframe: 6h, HTF: 1w. Target: 50-150 total trades over 4 years (12-37/year).

name = "6h_Donchian20_Breakout_1wEMA50_VolumeConfirm"
timeframe = "6h"
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
    
    # Calculate volume confirmation on 6h using 24-period MA (equivalent to 12h lookback)
    if len(volume) >= 24:
        vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
        volume_filter = volume > (1.8 * vol_ma_24)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    # Get 1w data ONCE before loop for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA50 trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 6h Donchian channels (20-period)
    if len(high) >= 20:
        # Donchian upper: 20-period high
        donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
        # Donchian lower: 20-period low
        donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    else:
        donchian_upper = np.full(n, np.nan)
        donchian_lower = np.full(n, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_in_trade = 0  # counter for time-based exit
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_in_trade = 0
            continue
        
        bars_in_trade += 1 if position != 0 else 0
        
        if position == 0:
            # Long conditions: price breaks above Donchian upper, weekly uptrend, volume filter
            if (close[i] > donchian_upper[i] and 
                open_price[i] <= donchian_upper[i] and  # Ensure breakout happens on this bar
                close[i] > ema_50_1w_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
                bars_in_trade = 1
            # Short conditions: price breaks below Donchian lower, weekly downtrend, volume filter
            elif (close[i] < donchian_lower[i] and 
                  open_price[i] >= donchian_lower[i] and  # Ensure breakdown happens on this bar
                  close[i] < ema_50_1w_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
                bars_in_trade = 1
        elif position != 0:
            # Time-based exit: close position after 3 bars (18h) to avoid whipsaw
            if bars_in_trade >= 3:
                signals[i] = 0.0
                position = 0
                bars_in_trade = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals