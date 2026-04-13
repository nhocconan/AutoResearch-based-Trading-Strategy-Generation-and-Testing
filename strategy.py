#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Donchian(20) breakout + weekly pivot direction + volume confirmation
    # Long when: price breaks above 20-period Donchian high AND weekly pivot > prior weekly close (bullish bias) AND volume > 1.5x average volume
    # Short when: price breaks below 20-period Donchian low AND weekly pivot < prior weekly close (bearish bias) AND volume > 1.5x average volume
    # Exit when: price crosses 10-period Donchian midpoint (mean reversion) OR volume drops below average
    # Uses discrete sizing (0.25) targeting 50-150 trades over 4 years.
    # Weekly pivot provides structural bias to avoid counter-trend trades in ranging markets.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot direction
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate weekly pivot points: P = (H + L + C)/3
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    # Weekly bias: bullish if pivot > prior week close, bearish if pivot < prior week close
    weekly_bullish = pivot_1w > close_1w  # pivot above weekly close = bullish
    weekly_bearish = pivot_1w < close_1w  # pivot below weekly close = bearish
    
    # Align weekly indicators to 6h timeframe
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish.astype(float))
    weekly_bearish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bearish.astype(float))
    
    # Calculate 6h Donchian channels
    lookback = 20
    # Donchian high: highest high over lookback period
    donchian_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    # Donchian low: lowest low over lookback period
    donchian_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    # Donchian midpoint: (high + low)/2
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Volume confirmation: volume > 1.5x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(vol_ma[i]) or
            np.isnan(weekly_bullish_aligned[i]) or np.isnan(weekly_bearish_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions
        long_breakout = close[i] > donchian_high[i-1]  # break above prior Donchian high
        short_breakout = close[i] < donchian_low[i-1]  # break below prior Donchian low
        
        long_entry = long_breakout and weekly_bullish_aligned[i] and vol_confirm[i]
        short_entry = short_breakout and weekly_bearish_aligned[i] and vol_confirm[i]
        
        # Exit conditions: price crosses Donchian midpoint OR volume drops below average
        exit_long = close[i] < donchian_mid[i] or not vol_confirm[i]
        exit_short = close[i] > donchian_mid[i] or not vol_confirm[i]
        
        # Entry logic
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        # Exit logic
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

name = "6h_1w_donchian_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0