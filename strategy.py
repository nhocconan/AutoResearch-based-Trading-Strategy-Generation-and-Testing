#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-hour strategy with 4-hour trend filter and 1-day volume confirmation
# Uses 4-hour Donchian(20) breakout for direction, confirmed by 1-day volume surge (>1.5x 20-day avg)
# Entry timing on 1-hour: price must close outside Donchian bands for confirmation
# Exit: reverse signal or stoploss at 2.5*ATR(14)
# Position size: 0.20 (20% of capital) to manage drawdown
# Session filter: 08:00-20:00 UTC to avoid low-volume periods
# Target: 60-150 total trades over 4 years (15-37/year)

name = "1h_donchian20_1d_vol_4h_trend_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4-hour data for trend filter (Donchian direction)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # 1-day data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 4-hour Donchian channels (20-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    highest_high_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lowest_low_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    highest_high_4h_aligned = align_htf_to_ltf(prices, df_4h, highest_high_4h)
    lowest_low_4h_aligned = align_htf_to_ltf(prices, df_4h, lowest_low_4h)
    
    # 1-day volume average (20-period)
    volume_1d = df_1d['volume'].values
    volume_1d_s = pd.Series(volume_1d)
    volume_ma_1d = volume_1d_s.rolling(window=20, min_periods=20).mean().values
    volume_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_1d)
    
    # 1-hour ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if required data not available or outside session
        if (np.isnan(highest_high_4h_aligned[i]) or np.isnan(lowest_low_4h_aligned[i]) or 
            np.isnan(volume_ma_1d_aligned[i]) or np.isnan(atr[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.5 * ATR
            if close[i] < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Reverse signal: price closes below 4h Donchian low
            elif close[i] < lowest_low_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            # Stoploss: 2.5 * ATR
            if close[i] > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Reverse signal: price closes above 4h Donchian high
            elif close[i] > highest_high_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.20
        else:
            # Look for entries: 4h Donchian breakout with 1d volume confirmation
            # Volume filter: current volume > 1.5x 20-day average
            volume_filter = volume[i] > 1.5 * volume_ma_1d_aligned[i]
            
            # Long: price closes above 4h Donchian high + volume filter
            if close[i] > highest_high_4h_aligned[i] and volume_filter:
                signals[i] = 0.20
                position = 1
                entry_price = close[i]
            # Short: price closes below 4h Donchian low + volume filter
            elif close[i] < lowest_low_4h_aligned[i] and volume_filter:
                signals[i] = -0.20
                position = -1
                entry_price = close[i]
    
    return signals