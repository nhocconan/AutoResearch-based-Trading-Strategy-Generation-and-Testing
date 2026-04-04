#!/usr/bin/env python3
"""
exp_6455_6h_weekly_pivot_donchian_vol_v1
Hypothesis: 6h Donchian(20) breakout with weekly pivot bias (above/below weekly pivot) and volume confirmation.
Weekly pivot provides structural bias from higher timeframe, Donchian captures breakouts,
volume confirms institutional participation. Works in bull (breakouts continuation) and bear (fade at pivot levels).
Target trades: 75-200 over 4 years (~19-50/year).
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_6455_6h_weekly_pivot_donchian_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate weekly pivot points (using prior week's OHLC)
    # Pivot = (H + L + C) / 3
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # Calculate 1d Donchian channels for trend filter (optional)
    # But we'll use 6h Donchian for entries
    
    # Calculate 6h Donchian channels
    lookback = 20
    high_roll = prices['high'].rolling(window=lookback, min_periods=lookback).max()
    low_roll = prices['low'].rolling(window=lookback, min_periods=lookback).min()
    donchian_upper = high_roll.shift(1)  # shift(1) to avoid look-ahead
    donchian_lower = low_roll.shift(1)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = prices['volume'].rolling(window=20, min_periods=20).mean()
    volume_filter = prices['volume'] > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # ATR for stoploss
    atr_period = 14
    tr1 = prices['high'] - prices['low']
    tr2 = abs(prices['high'] - prices['close'].shift(1))
    tr3 = abs(prices['low'] - prices['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=atr_period, min_periods=atr_period).mean()
    atr_values = atr.values
    
    for i in range(lookback, n):
        # Skip if volume filter not met
        if not volume_filter.iloc[i]:
            # If in position, check stoploss
            if position == 1 and prices['close'].iloc[i] < entry_price - 2.5 * atr_values[i]:
                signals[i] = 0.0
                position = 0
                continue
            elif position == -1 and prices['close'].iloc[i] > entry_price + 2.5 * atr_values[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = float(position * 0.25)  # maintain position size
            continue
            
        # Get current values
        close_price = prices['close'].iloc[i]
        upper = donchian_upper.iloc[i]
        lower = donchian_lower.iloc[i]
        pivot = weekly_pivot_aligned[i]
        
        # Long conditions: price breaks above Donchian upper AND above weekly pivot
        if close_price > upper and close_price > pivot:
            if position != 1:
                signals[i] = 0.25  # enter long 25%
                position = 1
                entry_price = close_price
            else:
                signals[i] = 0.25  # maintain
        # Short conditions: price breaks below Donchian lower AND below weekly pivot
        elif close_price < lower and close_price < pivot:
            if position != -1:
                signals[i] = -0.25  # enter short 25%
                position = -1
                entry_price = close_price
            else:
                signals[i] = -0.25  # maintain
        # Exit conditions: price returns to weekly pivot area
        elif position == 1 and close_price < pivot:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close_price > pivot:
            signals[i] = 0.0
            position = 0
        # Stoploss already handled above via volume filter check
        else:
            # Maintain current position
            signals[i] = float(position * 0.25)
    
    return signals