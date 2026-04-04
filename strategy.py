#!/usr/bin/env python3
"""
exp_6491_6h_donchian20_1d_pivot_vol_v1
Hypothesis: 6h Donchian(20) breakout with 1d Camarilla pivot levels and volume confirmation.
Uses 1d Camarilla pivot (R3/S3 for mean reversion, R4/S4 for breakout) to filter entries:
- Long when price breaks above Donchian HIGH AND above R4 (strong bullish continuation)
- Short when price breaks below Donchian LOW AND below S4 (strong bearish continuation)
Volume confirmation ensures breakouts have conviction.
Designed to work in both bull and bear markets by using Camarilla levels as dynamic support/resistance.
Target: 75-200 trades over 4 years (19-50/year) on 6h timeframe.
"""
from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_6491_6h_donchian20_1d_pivot_vol_v1"
timeframe = "6h"
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
    
    # Load HTF data ONCE before loop - using 1d for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    r4 = pivot + (range_1d * 1.1 / 2)
    r3 = pivot + (range_1d * 1.1 / 4)
    s3 = pivot - (range_1d * 1.1 / 4)
    s4 = pivot - (range_1d * 1.1 / 2)
    
    # Align to LTF (6h) with shift(1) for completed bars only
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
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
        if np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]):
            continue
            
        # Long conditions: price breaks above Donchian HIGH + above R4 + volume spike
        long_breakout = close[i] > donchian_high[i-1]  # break above previous period's high
        long_continuation = close[i] > r4_aligned[i]   # above R4 (strong bullish)
        long_volume = volume[i] > vol_ma[i] * VOL_THRESHOLD if not np.isnan(vol_ma[i]) else False
        
        # Short conditions: price breaks below Donchian LOW + below S4 + volume spike
        short_breakout = close[i] < donchian_low[i-1]  # break below previous period's low
        short_continuation = close[i] < s4_aligned[i]  # below S4 (strong bearish)
        short_volume = volume[i] > vol_ma[i] * VOL_THRESHOLD if not np.isnan(vol_ma[i]) else False
        
        # Exit conditions: midpoint reversal or opposite Camarilla level touch
        if position == 1:  # long position
            # Exit if price drops below midpoint of Donchian channel
            exit_long = close[i] < (donchian_high[i-1] + donchian_low[i-1]) / 2
            # Or if price touches S3 (strong reversal signal)
            exit_long = exit_long or close[i] < s3_aligned[i]
            if exit_long:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            # Exit if price rises above midpoint of Donchian channel
            exit_short = close[i] > (donchian_high[i-1] + donchian_low[i-1]) / 2
            # Or if price touches R3 (strong reversal signal)
            exit_short = exit_short or close[i] > r3_aligned[i]
            if exit_short:
                signals[i] = 0.0
                position = 0
                continue
        
        # Enter new positions only if flat
        if position == 0:
            if long_breakout and long_continuation and long_volume:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
            elif short_breakout and short_continuation and short_volume:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = position * SIGNAL_SIZE
    
    return signals