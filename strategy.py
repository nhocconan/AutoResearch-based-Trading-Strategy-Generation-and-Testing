#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d trend filter and volume confirmation
# Uses 1d EMA for trend direction to avoid counter-trend trades
# Volume filter ensures breakouts have conviction
# Target: 75-200 total trades over 4 years (19-50/year)
# Works in bull/bear by only trading with 1d trend
name = "4h_1d_donchian_trend_volume_v1"
timeframe = "4h"
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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA for trend filter
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate 1d average volume for confirmation (20-period)
    volume_1d = df_1d['volume'].values
    vol_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    
    # Calculate 4h Donchian channels (20-period)
    # Highest high and lowest low over past 20 periods
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start from index 60 to ensure sufficient data
    for i in range(60, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(vol_avg_20_1d_aligned[i]) or
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Current 1d volume (aligned)
        vol_1d_current = align_htf_to_ltf(prices, df_1d, volume_1d)[i]
        vol_confirm = vol_1d_current > vol_avg_20_1d_aligned[i]
        
        price = close[i]
        
        # Donchian breakout conditions with trend and volume filters
        # Long when price breaks above 20-period high AND above 1d EMA AND volume confirms
        long_breakout = price > highest_high[i-1]  # Previous bar's high to avoid look-ahead
        long_trend = price > ema_1d_aligned[i]
        long_signal = long_breakout and long_trend and vol_confirm
        
        # Short when price breaks below 20-period low AND below 1d EMA AND volume confirms
        short_breakout = price < lowest_low[i-1]   # Previous bar's low to avoid look-ahead
        short_trend = price < ema_1d_aligned[i]
        short_signal = short_breakout and short_trend and vol_confirm
        
        # Exit conditions
        exit_long = price < ema_1d_aligned[i]  # Exit when price crosses below 1d EMA
        exit_short = price > ema_1d_aligned[i]  # Exit when price crosses above 1d EMA
        
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