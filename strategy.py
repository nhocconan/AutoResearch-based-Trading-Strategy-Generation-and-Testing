#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation
# Long when price breaks above Donchian(20) upper band AND close > 1d EMA50 AND volume > 1.5x 20-period average
# Short when price breaks below Donchian(20) lower band AND close < 1d EMA50 AND volume > 1.5x 20-period average
# Exit when price crosses opposite Donchian band (lower for long exit, upper for short exit)
# Uses Donchian channels for breakout signals, effective in trending markets with volume confirmation.
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
    
    # Calculate volume confirmation on 12h (no HTF needed)
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
        # Upper band: highest high over last 20 periods
        upper_band = pd.Series(high).rolling(window=20, min_periods=20).max().values
        # Lower band: lowest low over last 20 periods
        lower_band = pd.Series(low).rolling(window=20, min_periods=20).min().values
    else:
        upper_band = np.full(n, np.nan)
        lower_band = np.full(n, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(upper_band[i]) or 
            np.isnan(lower_band[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above upper band AND above 1d EMA50 AND volume filter
            if (close[i] > upper_band[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below lower band AND below 1d EMA50 AND volume filter
            elif (close[i] < lower_band[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below lower band (breakdown below support)
            if close[i] < lower_band[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above upper band (breakout above resistance)
            if close[i] > upper_band[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals