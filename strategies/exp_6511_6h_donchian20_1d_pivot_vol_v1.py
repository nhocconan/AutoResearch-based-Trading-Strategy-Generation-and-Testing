#!/usr/bin/env python3
"""
exp_6511_6h_donchian20_1d_pivot_vol_v1
Hypothesis: 6h Donchian(20) breakout with 1d Camarilla pivot levels as directional filter and volume confirmation.
In bull markets: long when price > 1d pivot and breaks Donchian high with volume.
In bear markets: short when price < 1d pivot and breaks Donchian low with volume.
Uses 1d Camarilla pivots (calculated from prior 1d OHLC) to define institutional support/resistance.
Volume confirmation ensures breakouts have conviction. Designed for low-frequency, high-conviction trades.
Target: 75-150 total trades over 4 years (19-38/year) to minimize fee drag.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_6511_6h_donchian20_1d_pivot_vol_v1"
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
    
    # Calculate 1d Camarilla pivot levels (based on prior day's OHLC)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: based on previous day's range
    # R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low), etc.
    # S4 = close - 1.5*(high-low), S3 = close - 1.1*(high-low), etc.
    pivot = (high_1d + low_1d + close_1d) / 3.0
    range_hl = high_1d - low_1d
    
    # Key levels: R3, S3 for fade; R4, S4 for breakout
    r3 = close_1d + 1.1 * range_hl
    s3 = close_1d - 1.1 * range_hl
    r4 = close_1d + 1.5 * range_hl
    s4 = close_1d - 1.5 * range_hl
    
    # Align to LTF (6h) with shift(1) for completed bars only
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
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
        # Skip if HTF data not available
        if np.isnan(pivot_aligned[i]):
            continue
            
        # Long conditions: price > pivot (bullish bias) + breaks above Donchian HIGH + volume spike
        long_bias = close[i] > pivot_aligned[i]  # price above daily pivot (bullish)
        long_breakout = close[i] > donchian_high[i-1]  # break above previous period's high
        long_volume = volume[i] > vol_ma[i] * VOL_THRESHOLD if not np.isnan(vol_ma[i]) else False
        
        # Short conditions: price < pivot (bearish bias) + breaks below Donchian LOW + volume spike
        short_bias = close[i] < pivot_aligned[i]  # price below daily pivot (bearish)
        short_breakout = close[i] < donchian_low[i-1]  # break below previous period's low
        short_volume = volume[i] > vol_ma[i] * VOL_THRESHOLD if not np.isnan(vol_ma[i]) else False
        
        # Exit conditions: pivot reversal or midpoint reversal
        if position == 1:  # long position
            # Exit if price drops back below pivot (trend change)
            exit_long = close[i] < pivot_aligned[i]
            # Or if price drops below midpoint of channel
            exit_long = exit_long or close[i] < (donchian_high[i-1] + donchian_low[i-1]) / 2
            if exit_long:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            # Exit if price rises back above pivot (trend change)
            exit_short = close[i] > pivot_aligned[i]
            # Or if price rises above midpoint of channel
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