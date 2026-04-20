#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d trend filter, volume confirmation, and ATR-based stop
# Enters long when price breaks above 20-period high with daily uptrend (close > EMA50) and volume > 1.5x average.
# Enters short when price breaks below 20-period low with daily downtrend (close < EMA50) and volume > 1.5x average.
# Exits when price crosses the 20-period moving average in opposite direction.
# Uses tight entry conditions to limit trades (~30-50/year) and avoid fee drag.
# Trend filter prevents counter-trend trades, volume confirms institutional interest.

name = "4h_Donchian20_1dEMA50_Volume_ATR"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # === Donchian Channels (20-period high/low) ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # 20-period high and low for Donchian
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    mid_20 = (high_20 + low_20) / 2.0  # 20-period midpoint for exit
    
    # === Daily EMA50 for trend filter ===
    daily_close = df_1d['close'].values
    ema_50 = pd.Series(daily_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # === Volume confirmation ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values  # 20-period average
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, np.nan)
    
    # === ATR for stop loss (optional, using signal-based exit instead) ===
    # Using price crossing mid-point as exit mechanism instead of ATR stop
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after warmup
        # Get values
        close_val = close[i]
        high_20_val = high_20[i]
        low_20_val = low_20[i]
        mid_20_val = mid_20[i]
        ema_val = ema_50_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(high_20_val) or np.isnan(low_20_val) or np.isnan(mid_20_val) or 
            np.isnan(ema_val) or np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: price breaks above 20-period high, daily uptrend, volume confirmation
            if close_val > high_20_val and close_val > ema_val and vol_ratio_val > 1.5:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below 20-period low, daily downtrend, volume confirmation
            elif close_val < low_20_val and close_val < ema_val and vol_ratio_val > 1.5:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below 20-period midpoint or trend breaks
            if close_val < mid_20_val or close_val < ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above 20-period midpoint or trend breaks
            if close_val > mid_20_val or close_val > ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals