#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1h Camarilla breakout with 4h trend filter and volume confirmation
    # Long when: price breaks above 1h Donchian(20) high AND price > 4h Camarilla H3 AND volume > 1.5x avg volume AND 4h close > 4h open (bullish candle)
    # Short when: price breaks below 1h Donchian(20) low AND price < 4h Camarilla L3 AND volume > 1.5x avg volume AND 4h close < 4h open (bearish candle)
    # Exit when: price crosses 1h Donchian midpoint
    # Uses 4h for trend bias and Camarilla structure, 1h for precise entry timing.
    # Session filter: 08-20 UTC to avoid low-volume periods.
    # Discrete sizing: 0.20 targeting 60-150 total trades over 4 years.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute hour for session filter
    hours = pd.DatetimeIndex(open_time).hour
    
    # Get 4h data for trend and Camarilla
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    open_4h = df_4h['open'].values
    
    # Calculate 4h Camarilla pivots (using previous bar's range)
    range_4h = high_4h - low_4h
    h3_4h = close_4h + 1.125 * range_4h
    l3_4h = close_4h - 1.125 * range_4h
    
    # Align 4h Camarilla levels to 1h timeframe
    h3_4h_aligned = align_htf_to_ltf(prices, df_4h, h3_4h)
    l3_4h_aligned = align_htf_to_ltf(prices, df_4h, l3_4h)
    
    # Calculate 4h bullish/bearish candle filter
    bullish_4h = close_4h > open_4h
    bearish_4h = close_4h < open_4h
    bullish_4h_aligned = align_htf_to_ltf(prices, df_4h, bullish_4h.astype(float))
    bearish_4h_aligned = align_htf_to_ltf(prices, df_4h, bearish_4h.astype(float))
    
    # Calculate Donchian(20) channels on 1h
    lookback = 20
    donchian_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    donchian_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_threshold = vol_ma * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.20  # 20% position size
    
    for i in range(100, n):
        # Session filter: 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(h3_4h_aligned[i]) or np.isnan(l3_4h_aligned[i]) or
            np.isnan(vol_threshold[i]) or np.isnan(bullish_4h_aligned[i]) or
            np.isnan(bearish_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_ok = volume[i] > vol_threshold[i]
        
        # Breakout conditions
        long_breakout = close[i] > donchian_high[i]
        short_breakout = close[i] < donchian_low[i]
        
        # Camarilla and trend filters
        long_filter = close[i] > h3_4h_aligned[i] and bullish_4h_aligned[i] > 0.5
        short_filter = close[i] < l3_4h_aligned[i] and bearish_4h_aligned[i] > 0.5
        
        # Entry conditions
        long_entry = long_breakout and long_filter and vol_ok and position != 1
        short_entry = short_breakout and short_filter and vol_ok and position != -1
        
        # Exit conditions: price crosses Donchian midpoint
        exit_long = close[i] < donchian_mid[i]
        exit_short = close[i] > donchian_mid[i]
        
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

name = "1h_4h_camarilla_breakout_volume_session_v1"
timeframe = "1h"
leverage = 1.0