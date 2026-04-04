#!/usr/bin/env python3
"""
exp_6555_6h_donchian20_1w_pivot_vol_v1
Hypothesis: 6h Donchian(20) breakout with weekly pivot direction and volume confirmation.
Weekly pivot (calculated from prior week's OHLC) provides structural bias: long above weekly pivot, short below.
Donchian breakout confirms momentum in direction of weekly bias. Volume spike (2.0x 20-period MA) validates breakout.
Designed for 75-150 total trades over 4 years with discrete sizing (0.25) to minimize fee drag.
Works in bull/bear: weekly pivot adapts to longer-term structure, avoiding whipsaws in ranging markets.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_6555_6h_donchian20_1w_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOL_MA_PERIOD = 20
VOL_THRESHOLD = 2.0     # volume must be 2.0x its 20-period MA
SIGNAL_SIZE = 0.25      # 25% position size

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 1w for weekly pivot
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot points: P = (H+L+C)/3
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    weekly_pivot = (high_1w + low_1w + close_1w) / 3.0
    
    # Align to LTF (6h) with shift(1) for completed bars only
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
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
        # Skip if HTF data not available
        if np.isnan(weekly_pivot_aligned[i]):
            continue
            
        # Long conditions: price > weekly pivot (bullish bias) + breaks above Donchian HIGH + volume spike
        long_bias = close[i] > weekly_pivot_aligned[i]  # price above weekly pivot
        long_breakout = close[i] > donchian_high[i-1]  # break above previous period's high
        long_volume = volume[i] > vol_ma[i] * VOL_THRESHOLD if not np.isnan(vol_ma[i]) else False
        
        # Short conditions: price < weekly pivot (bearish bias) + breaks below Donchian LOW + volume spike
        short_bias = close[i] < weekly_pivot_aligned[i]  # price below weekly pivot
        short_breakout = close[i] < donchian_low[i-1]  # break below previous period's low
        short_volume = volume[i] > vol_ma[i] * VOL_THRESHOLD if not np.isnan(vol_ma[i]) else False
        
        # Exit conditions: weekly pivot reversal or Donchian midpoint reversal
        if position == 1:  # long position
            # Exit if price drops back below weekly pivot (bias change)
            exit_long = close[i] < weekly_pivot_aligned[i]
            # Or if price drops below Donchian midpoint
            exit_long = exit_long or close[i] < (donchian_high[i-1] + donchian_low[i-1]) / 2
            if exit_long:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            # Exit if price rises back above weekly pivot (bias change)
            exit_short = close[i] > weekly_pivot_aligned[i]
            # Or if price rises above Donchian midpoint
            exit_short = exit_short or close[i] > (donchian_high[i-1] + donchian_low[i-1]) / 2
            if exit_short:
                signals[i] = 0.0
                position = 0
                continue
        
        # Enter new positions only if flat
        if position == 0:
            if long_bias and long_breakout and long_volume:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
            elif short_bias and short_breakout and short_volume:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = position * SIGNAL_SIZE
    
    return signals