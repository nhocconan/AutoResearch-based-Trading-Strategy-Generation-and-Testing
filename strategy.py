#!/usr/bin/env python3
"""
exp_6499_6h_donchian20_12h_1d_pivot_v1
Hypothesis: 6h Donchian(20) breakout with 12h pivot point direction filter and 1d volume confirmation.
Uses daily pivot points calculated from prior 1d OHLC: long when price > pivot, short when price < pivot.
Donchian(20) breakout provides entry timing, volume confirmation filters weak breakouts.
Designed to work in both bull and bear markets by using 12h pivot as trend filter and Donchian breakouts for momentum.
Target: 75-150 trades over 4 years (19-38/year) to stay within profitable range while ensuring statistical validity.
"""
from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_6499_6h_donchian20_12h_1d_pivot_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
PIVOT_LOOKBACK = 1  # use prior day's OHLC for pivot
VOL_MA_PERIOD = 20
VOL_THRESHOLD = 1.8
SIGNAL_SIZE = 0.25   # 25% position size

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 12h for pivot and 1d for volume MA
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 12h pivot points from prior 12h bar's OHLC (shifted by 1 for completed bar)
    # Pivot = (High + Low + Close) / 3
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    pivot_12h = (high_12h + low_12h + close_12h) / 3.0
    
    # Align to LTF (6h) with shift(1) for completed bars only
    pivot_12h_aligned = align_htf_to_ltf(prices, df_12h, pivot_12h)
    
    # Calculate 1d volume MA for confirmation
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=VOL_MA_PERIOD, min_periods=VOL_MA_PERIOD, adjust=False).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    donchian_low = pd.Series(low).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, VOL_MA_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if pivot or volume MA data not available
        if np.isnan(pivot_12h_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]):
            continue
            
        # Long conditions: price breaks above Donchian HIGH + above 12h pivot + volume spike
        long_breakout = close[i] > donchian_high[i-1]  # break above previous period's high
        long_trend = close[i] > pivot_12h_aligned[i]   # price above 12h pivot (bullish bias)
        long_volume = volume[i] > vol_ma_1d_aligned[i] * VOL_THRESHOLD if not np.isnan(vol_ma_1d_aligned[i]) else False
        
        # Short conditions: price breaks below Donchian LOW + below 12h pivot + volume spike
        short_breakout = close[i] < donchian_low[i-1]  # break below previous period's low
        short_trend = close[i] < pivot_12h_aligned[i]  # price below 12h pivot (bearish bias)
        short_volume = volume[i] > vol_ma_1d_aligned[i] * VOL_THRESHOLD if not np.isnan(vol_ma_1d_aligned[i]) else False
        
        # Exit conditions: midpoint reversal or opposite pivot break
        if position == 1:  # long position
            # Exit if price drops below midpoint of channel
            exit_long = close[i] < (donchian_high[i-1] + donchian_low[i-1]) / 2
            # Or if price breaks below Donchian low (strong reversal)
            exit_long = exit_long or close[i] < donchian_low[i-1]
            # Or if price crosses below 12h pivot (trend change)
            exit_long = exit_long or close[i] < pivot_12h_aligned[i]
            if exit_long:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            # Exit if price rises above midpoint of channel
            exit_short = close[i] > (donchian_high[i-1] + donchian_low[i-1]) / 2
            # Or if price breaks above Donchian high (strong reversal)
            exit_short = exit_short or close[i] > donchian_high[i-1]
            # Or if price crosses above 12h pivot (trend change)
            exit_short = exit_short or close[i] > pivot_12h_aligned[i]
            if exit_short:
                signals[i] = 0.0
                position = 0
                continue
        
        # Enter new positions only if flat
        if position == 0:
            if long_breakout and long_trend and long_volume:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
            elif short_breakout and short_trend and short_volume:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = position * SIGNAL_SIZE
    
    return signals