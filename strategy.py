#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams %R
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate Williams %R (14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    highest_high = np.full(len(high_1d), np.nan)
    lowest_low = np.full(len(low_1d), np.nan)
    for i in range(14, len(high_1d)):
        highest_high[i] = np.max(high_1d[i-14:i])
        lowest_low[i] = np.min(low_1d[i-14:i])
    
    williams_r = np.full(len(high_1d), np.nan)
    for i in range(14, len(high_1d)):
        denominator = highest_high[i] - lowest_low[i]
        if denominator != 0:
            williams_r[i] = (highest_high[i] - close_1d[i]) / denominator * -100
        else:
            williams_r[i] = 0
    
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Get 1d data for volume filter
    vol_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 21:
        return np.zeros(n)
    
    # Calculate 4h EMA(21)
    close_4h = df_4h['close'].values
    ema_21_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_21_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Warmup: need Williams %R, volume MA, and EMA
    start_idx = max(14, 20, 21)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(williams_r_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i]) or 
            np.isnan(ema_21_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        wr = williams_r_aligned[i]
        vol_now = volume[i]
        vol_ma = vol_ma_20_1d_aligned[i]
        ema_trend = ema_21_4h_aligned[i]
        
        # Volume filter: volume > 1.3x 1d MA
        vol_breakout = vol_now > 1.3 * vol_ma
        
        # Entry conditions
        if position == 0:
            # Long: Williams %R oversold (< -80) + price above EMA + volume breakout
            if wr < -80 and close[i] > ema_trend and vol_breakout:
                signals[i] = size
                position = 1
            # Short: Williams %R overbought (> -20) + price below EMA + volume breakout
            elif wr > -20 and close[i] < ema_trend and vol_breakout:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Williams %R overbought (> -20) or volume dries up
            if wr > -20 or vol_now < 0.7 * vol_ma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: Williams %R oversold (< -80) or volume dries up
            if wr < -80 or vol_now < 0.7 * vol_ma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_WilliamsR_VolumeBreakout_EMA21Trend"
timeframe = "4h"
leverage = 1.0