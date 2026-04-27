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
    
    # Get 1d data for Donchian channel
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d Donchian(20)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    upper = np.full(len(high_1d), np.nan)
    lower = np.full(len(high_1d), np.nan)
    for i in range(20, len(high_1d)):
        upper[i] = np.max(high_1d[i-20:i])
        lower[i] = np.min(low_1d[i-20:i])
    donch_upper_1d = upper
    donch_lower_1d = lower
    donch_upper_1d_aligned = align_htf_to_ltf(prices, df_1d, donch_upper_1d)
    donch_lower_1d_aligned = align_htf_to_ltf(prices, df_1d, donch_lower_1d)
    
    # Get 1w data for weekly EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA(50)
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Get 1d data for volume filter
    vol_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Warmup: need Donchian, EMA, and volume MA
    start_idx = max(20, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donch_upper_1d_aligned[i]) or np.isnan(donch_lower_1d_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        upper = donch_upper_1d_aligned[i]
        lower = donch_lower_1d_aligned[i]
        ema_trend = ema_50_1w_aligned[i]
        vol_now = volume[i]
        vol_ma = vol_ma_20_1d_aligned[i]
        
        # Volume filter: volume > 1.5x 1d MA (volume breakout)
        vol_breakout = vol_now > 1.5 * vol_ma
        
        # Entry conditions: breakout with volume and trend filter
        if position == 0:
            # Long: break above upper band + volume + price above weekly EMA
            if close[i] > upper and vol_breakout and close[i] > ema_trend:
                signals[i] = size
                position = 1
            # Short: break below lower band + volume + price below weekly EMA
            elif close[i] < lower and vol_breakout and close[i] < ema_trend:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: close below lower band or trend reversal
            if close[i] < lower or close[i] < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: close above upper band or trend reversal
            if close[i] > upper or close[i] > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_Donchian20_WeeklyEMA50_VolumeBreakout"
timeframe = "1d"
leverage = 1.0