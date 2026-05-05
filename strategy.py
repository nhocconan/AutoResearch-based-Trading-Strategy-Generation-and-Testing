#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + volume confirmation + 1d EMA50 trend filter
# Long when price breaks above Donchian(20) high AND volume > 1.5x 20-period average AND close > 1d EMA50
# Short when price breaks below Donchian(20) low AND volume > 1.5x 20-period average AND close < 1d EMA50
# Exit when price returns to Donchian(20) midpoint (mean reversion within the channel)
# Uses Donchian channels for breakout structure, effective in both bull (continuation) and bear (mean reversion via exits) markets.
# Timeframe: 12h, HTF: 1d. Target: 50-150 total trades over 4 years (12-37/year).

name = "12h_Donchian20_Breakout_1dEMA50_Volume"
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
    
    # Calculate volume confirmation on 12h
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    # Get 1d data ONCE before loop for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Donchian(20) channels on 12h
    if len(high) >= 20 and len(low) >= 20:
        # Donchian high: highest high over past 20 periods
        dc_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
        # Donchian low: lowest low over past 20 periods
        dc_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
        # Donchian midpoint: average of high and low
        dc_mid = (dc_high + dc_low) / 2.0
    else:
        dc_high = np.full(n, np.nan)
        dc_low = np.full(n, np.nan)
        dc_mid = np.full(n, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(dc_high[i]) or 
            np.isnan(dc_low[i]) or 
            np.isnan(dc_mid[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian high AND volume filter AND above 1d EMA50
            if (close[i] > dc_high[i] and 
                volume_filter[i] and 
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian low AND volume filter AND below 1d EMA50
            elif (close[i] < dc_low[i] and 
                  volume_filter[i] and 
                  close[i] < ema_50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to Donchian midpoint (mean reversion)
            if close[i] <= dc_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to Donchian midpoint (mean reversion)
            if close[i] >= dc_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals