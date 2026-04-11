#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian breakout with 12h trend filter and volume confirmation
# Uses 12h EMA for trend direction to avoid counter-trend trades
# Volume filter ensures breakouts have conviction
# Target: 50-150 total trades over 4 years (12-37/year)
# Works in bull/bear by only trading with 12h trend
name = "6h_12h_donchian_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA for trend filter
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Calculate 6-period average 12h volume for confirmation
    volume_12h = df_12h['volume'].values
    vol_avg_6_12h = pd.Series(volume_12h).rolling(window=6, min_periods=6).mean().values
    vol_avg_6_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_avg_6_12h)
    
    # Calculate 6h Donchian channels (20-period)
    # Highest high and lowest low over past 20 periods
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start from index 60 to ensure sufficient data
    for i in range(60, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_12h_aligned[i]) or np.isnan(vol_avg_6_12h_aligned[i]) or
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Current 12h volume (aligned)
        vol_12h_current = align_htf_to_ltf(prices, df_12h, volume_12h)[i]
        vol_confirm = vol_12h_current > vol_avg_6_12h_aligned[i]
        
        price = close[i]
        
        # Donchian breakout conditions with trend and volume filters
        # Long when price breaks above 20-period high AND above 12h EMA AND volume confirms
        long_breakout = price > highest_high[i-1]  # Previous bar's high to avoid look-ahead
        long_trend = price > ema_12h_aligned[i]
        long_signal = long_breakout and long_trend and vol_confirm
        
        # Short when price breaks below 20-period low AND below 12h EMA AND volume confirms
        short_breakout = price < lowest_low[i-1]   # Previous bar's low to avoid look-ahead
        short_trend = price < ema_12h_aligned[i]
        short_signal = short_breakout and short_trend and vol_confirm
        
        # Exit conditions
        exit_long = price < ema_12h_aligned[i]  # Exit when price crosses below 12h EMA
        exit_short = price > ema_12h_aligned[i]  # Exit when price crosses above 12h EMA
        
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals