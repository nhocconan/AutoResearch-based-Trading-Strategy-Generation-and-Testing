#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Donchian(20) breakout + 12h Camarilla pivot structure + volume confirmation
    # Long when: price breaks above Donchian(20) high AND price > Camarilla H3 (12h) AND volume > 1.5x avg volume
    # Short when: price breaks below Donchian(20) low AND price < Camarilla L3 (12h) AND volume > 1.5x avg volume
    # Exit when: price crosses Donchian midpoint OR volume drops below average
    # Uses discrete sizing (0.25) targeting 50-150 trades over 4 years.
    # Works in bull/bear via Camarilla pivot structure providing dynamic support/resistance levels.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Camarilla pivots
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h Camarilla pivots (using previous day's range)
    # Camarilla levels: H4 = close + 1.5*(high-low), H3 = close + 1.125*(high-low)
    #                 L3 = close - 1.125*(high-low), L4 = close - 1.5*(high-low)
    # We'll use H3/L3 as entry filters and H4/L4 as stop levels
    range_12h = high_12h - low_12h
    h3_12h = close_12h + 1.125 * range_12h
    l3_12h = close_12h - 1.125 * range_12h
    h4_12h = close_12h + 1.5 * range_12h
    l4_12h = close_12h - 1.5 * range_12h
    
    # Align 12h Camarilla levels to 6h timeframe
    h3_12h_aligned = align_htf_to_ltf(prices, df_12h, h3_12h)
    l3_12h_aligned = align_htf_to_ltf(prices, df_12h, l3_12h)
    h4_12h_aligned = align_htf_to_ltf(prices, df_12h, h4_12h)
    l4_12h_aligned = align_htf_to_ltf(prices, df_12h, l4_12h)
    
    # Calculate Donchian(20) channels on 6h
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
            np.isnan(h3_12h_aligned[i]) or np.isnan(l3_12h_aligned[i]) or
            np.isnan(vol_threshold[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_ok = volume[i] > vol_threshold[i]
        
        # Breakout conditions
        long_breakout = close[i] > donchian_high[i]
        short_breakout = close[i] < donchian_low[i]
        
        # Camarilla filters
        long_filter = close[i] > h3_12h_aligned[i]
        short_filter = close[i] < l3_12h_aligned[i]
        
        # Entry conditions
        long_entry = long_breakout and long_filter and vol_ok and position != 1
        short_entry = short_breakout and short_filter and vol_ok and position != -1
        
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

name = "6h_12h_donchian_camarilla_volume_v1"
timeframe = "6h"
leverage = 1.0