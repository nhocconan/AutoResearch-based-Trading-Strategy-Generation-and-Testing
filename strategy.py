#!/usr/bin/env python3
"""
exp_6482_12h_donchian20_1d_pivot_vol_v1
Hypothesis: 12h Donchian(20) breakout with daily pivot bias and volume confirmation.
Uses daily pivot points (P, R1, S1) to determine structural bias: long when price > daily pivot and breaks above Donchian high, short when price < daily pivot and breaks below Donchian low.
Volume confirmation filters weak breakouts. Designed to work in both bull and bear markets by using daily pivot as structural reference and Donchian breakouts for momentum entries.
Target: 50-150 trades over 4 years (12-37/year). Uses 12h primary timeframe per experiment instructions.
"""
from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_6482_12h_donchian20_1d_pivot_vol_v1"
timeframe = "12h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOL_MA_PERIOD = 20
VOL_THRESHOLD = 1.8  # volume must be 1.8x its 20-period MA
SIGNAL_SIZE = 0.25   # 25% position size

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 1d for daily pivots
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily pivot points
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Daily pivot point: (high + low + close) / 3
    daily_pivot = (high_1d + low_1d + close_1d) / 3.0
    # Daily support/resistance levels
    daily_r1 = 2 * daily_pivot - low_1d
    daily_s1 = 2 * daily_pivot - high_1d
    
    # Align to LTF (12h) with shift(1) for completed bars only
    daily_pivot_aligned = align_htf_to_ltf(prices, df_1d, daily_pivot)
    daily_r1_aligned = align_htf_to_ltf(prices, df_1d, daily_r1)
    daily_s1_aligned = align_htf_to_ltf(prices, df_1d, daily_s1)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    donchian_low = pd.Series(low).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
    # Volume MA for confirmation
    vol_ma = pd.Series(volume).rolling(window=VOL_MA_PERIOD, min_periods=VOL_MA_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, VOL_MA_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if pivot data not available
        if np.isnan(daily_pivot_aligned[i]):
            continue
            
        # Long conditions: price breaks above Donchian HIGH + above daily pivot + volume spike
        long_breakout = close[i] > donchian_high[i-1]  # break above previous period's high
        long_bias = close[i] > daily_pivot_aligned[i]  # price above daily pivot (bullish bias)
        long_volume = volume[i] > vol_ma[i] * VOL_THRESHOLD if not np.isnan(vol_ma[i]) else False
        
        # Short conditions: price breaks below Donchian LOW + below daily pivot + volume spike
        short_breakout = close[i] < donchian_low[i-1]  # break below previous period's low
        short_bias = close[i] < daily_pivot_aligned[i]  # price below daily pivot (bearish bias)
        short_volume = volume[i] > vol_ma[i] * VOL_THRESHOLD if not np.isnan(vol_ma[i]) else False
        
        # Exit conditions: simple midpoint reversal
        channel_width = donchian_high[i-1] - donchian_low[i-1]
        if position == 1:  # long position
            # Exit if price drops below midpoint of channel
            exit_long = close[i] < (donchian_high[i-1] + donchian_low[i-1]) / 2
            # Or if price breaks below Donchian low (strong reversal)
            exit_long = exit_long or close[i] < donchian_low[i-1]
            if exit_long:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            # Exit if price rises above midpoint of channel
            exit_short = close[i] > (donchian_high[i-1] + donchian_low[i-1]) / 2
            # Or if price breaks above Donchian high (strong reversal)
            exit_short = exit_short or close[i] > donchian_high[i-1]
            if exit_short:
                signals[i] = 0.0
                position = 0
                continue
        
        # Enter new positions only if flat
        if position == 0:
            if long_breakout and long_bias and long_volume:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
            elif short_breakout and short_bias and short_volume:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = position * SIGNAL_SIZE
    
    return signals