#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h Donchian(20) breakout + 1w EMA(50) trend + volume confirmation
    # Long when: price breaks above Donchian(20) high AND price > 1w EMA(50) AND volume > 1.5x avg volume
    # Short when: price breaks below Donchian(20) low AND price < 1w EMA(50) AND volume > 1.5x avg volume
    # Exit when: price crosses Donchian midpoint OR volume drops below average
    # Uses discrete sizing (0.25) targeting 50-150 trades over 4 years.
    # Works in bull/bear via 1w EMA providing major trend filter and Donchian providing structure.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA(50) trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Calculate 1w EMA(50)
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Align 1w EMA to 12h timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Get 1d data for additional context (optional filter)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Calculate 1d EMA(20) for additional trend confirmation
    ema_20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    # Calculate Donchian(20) channels on 12h
    lookback = 20
    donchian_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    donchian_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_threshold = vol_ma * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(ema_20_1d_aligned[i]) or
            np.isnan(vol_threshold[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_ok = volume[i] > vol_threshold[i]
        
        # Breakout conditions
        long_breakout = close[i] > donchian_high[i]
        short_breakout = close[i] < donchian_low[i]
        
        # Trend filters: price above both 1w EMA(50) and 1d EMA(20) for long
        # price below both for short
        long_trend = close[i] > ema_50_1w_aligned[i] and close[i] > ema_20_1d_aligned[i]
        short_trend = close[i] < ema_50_1w_aligned[i] and close[i] < ema_20_1d_aligned[i]
        
        # Entry conditions
        long_entry = long_breakout and long_trend and vol_ok and position != 1
        short_entry = short_breakout and short_trend and vol_ok and position != -1
        
        # Exit conditions: price crosses Donchian midpoint OR volume drops below average
        exit_long = close[i] < donchian_mid[i] or volume[i] < vol_ma[i]
        exit_short = close[i] > donchian_mid[i] or volume[i] < vol_ma[i]
        
        # Execute signals
        if long_entry:
            position = 1
            signals[i] = position_size
        elif short_entry:
            position = -1
            signals[i] = -position_size
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_1w_donchian_ema_volume_v1"
timeframe = "12h"
leverage = 1.0