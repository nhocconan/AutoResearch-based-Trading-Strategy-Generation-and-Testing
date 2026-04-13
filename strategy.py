#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Donchian(20) breakout with 1d weekly pivot structure
    # Long when: price breaks above Donchian(20) high AND price > weekly H3 (1d) AND volume > 1.5x avg volume
    # Short when: price breaks below Donchian(20) low AND price < weekly L3 (1d) AND volume > 1.5x avg volume
    # Exit when: price crosses Donchian midpoint
    # Uses discrete sizing (0.25) targeting 50-150 trades over 4 years.
    # Weekly pivots from 1d data provide dynamic structure that works in both bull/bear markets.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for weekly Camarilla pivots (using last 5 days to approximate weekly)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 5-day (weekly) Camarilla pivots using rolling max/min
    lookback_5d = 5
    weekly_high = pd.Series(high_1d).rolling(window=lookback_5d, min_periods=lookback_5d).max().values
    weekly_low = pd.Series(low_1d).rolling(window=lookback_5d, min_periods=lookback_5d).min().values
    weekly_close = pd.Series(close_1d).rolling(window=lookback_5d, min_periods=lookback_5d).last().values
    
    # Weekly Camarilla levels
    weekly_range = weekly_high - weekly_low
    h3_1d = weekly_close + 1.125 * weekly_range
    l3_1d = weekly_close - 1.125 * weekly_range
    h4_1d = weekly_close + 1.5 * weekly_range
    l4_1d = weekly_close - 1.5 * weekly_range
    
    # Align 1d weekly Camarilla levels to 6h timeframe
    h3_1d_aligned = align_htf_to_ltf(prices, df_1d, h3_1d)
    l3_1d_aligned = align_htf_to_ltf(prices, df_1d, l3_1d)
    h4_1d_aligned = align_htf_to_ltf(prices, df_1d, h4_1d)
    l4_1d_aligned = align_htf_to_ltf(prices, df_1d, l4_1d)
    
    # Calculate Donchian(20) channels on 6h
    lookback_dc = 20
    donchian_high = pd.Series(high).rolling(window=lookback_dc, min_periods=lookback_dc).max().values
    donchian_low = pd.Series(low).rolling(window=lookback_dc, min_periods=lookback_dc).min().values
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
            np.isnan(h3_1d_aligned[i]) or np.isnan(l3_1d_aligned[i]) or
            np.isnan(vol_threshold[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_ok = volume[i] > vol_threshold[i]
        
        # Breakout conditions
        long_breakout = close[i] > donchian_high[i]
        short_breakout = close[i] < donchian_low[i]
        
        # Weekly Camarilla filters
        long_filter = close[i] > h3_1d_aligned[i]
        short_filter = close[i] < l3_1d_aligned[i]
        
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

name = "6h_1d_weekly_camarilla_donchian_volume_v1"
timeframe = "6h"
leverage = 1.0