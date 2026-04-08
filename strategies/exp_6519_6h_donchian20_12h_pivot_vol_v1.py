#!/usr/bin/env python3
"""
exp_6519_6h_donchian20_12h_pivot_vol_v1
Hypothesis: 6h Donchian(20) breakout with 12h Camarilla pivot levels (R4/S4) as breakout confirmation and volume filter.
In bull markets: long when price > 12h EMA50 and breaks above Donchian high with volume > 1.5x MA.
In bear markets: short when price < 12h EMA50 and breaks below Donchian low with volume > 1.5x MA.
Uses 12h Camarilla R4/S4 levels to avoid false breakouts (only trade breakouts beyond these extreme levels).
Designed for low-frequency, high-conviction trades targeting 50-150 total trades over 4 years.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_6519_6h_donchian20_12h_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
EMA_PERIOD = 50
VOL_MA_PERIOD = 20
VOL_THRESHOLD = 1.5  # volume must be 1.5x its 20-period MA
SIGNAL_SIZE = 0.25   # 25% position size

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 12h for EMA50 and Camarilla pivots
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h EMA50
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=EMA_PERIOD, adjust=False).mean().values
    
    # Calculate 12h Camarilla pivot levels (R4, S4)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    pivot_12h = (high_12h + low_12h + close_12h) / 3
    range_12h = high_12h - low_12h
    r4_12h = pivot_12h + (range_12h * 1.5)  # R4 = pivot + 1.5*range
    s4_12h = pivot_12h - (range_12h * 1.5)  # S4 = pivot - 1.5*range
    
    # Align to LTF (6h) with shift(1) for completed bars only
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    r4_12h_aligned = align_htf_to_ltf(prices, df_12h, r4_12h)
    s4_12h_aligned = align_htf_to_ltf(prices, df_12h, s4_12h)
    
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
        if np.isnan(ema_12h_aligned[i]) or np.isnan(r4_12h_aligned[i]) or np.isnan(s4_12h_aligned[i]):
            continue
            
        # Long conditions: price > 12h EMA50 (bullish bias) + breaks above Donchian HIGH + breaks above R4 + volume spike
        long_bias = close[i] > ema_12h_aligned[i]  # price above 12h EMA50 (bullish)
        long_breakout = close[i] > donchian_high[i-1]  # break above previous period's high
        long_pivot = close[i] > r4_12h_aligned[i]  # break above Camarilla R4 (strong breakout)
        long_volume = volume[i] > vol_ma[i] * VOL_THRESHOLD if not np.isnan(vol_ma[i]) else False
        
        # Short conditions: price < 12h EMA50 (bearish bias) + breaks below Donchian LOW + breaks below S4 + volume spike
        short_bias = close[i] < ema_12h_aligned[i]  # price below 12h EMA50 (bearish)
        short_breakout = close[i] < donchian_low[i-1]  # break below previous period's low
        short_pivot = close[i] < s4_12h_aligned[i]  # break below Camarilla S4 (strong breakout)
        short_volume = volume[i] > vol_ma[i] * VOL_THRESHOLD if not np.isnan(vol_ma[i]) else False
        
        # Exit conditions: EMA reversal or midpoint reversal
        if position == 1:  # long position
            # Exit if price drops back below EMA50 (trend change)
            exit_long = close[i] < ema_12h_aligned[i]
            # Or if price drops below midpoint of channel
            exit_long = exit_long or close[i] < (donchian_high[i-1] + donchian_low[i-1]) / 2
            if exit_long:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            # Exit if price rises back above EMA50 (trend change)
            exit_short = close[i] > ema_12h_aligned[i]
            # Or if price rises above midpoint of channel
            exit_short = exit_short or close[i] > (donchian_high[i-1] + donchian_low[i-1]) / 2
            if exit_short:
                signals[i] = 0.0
                position = 0
                continue
        
        # Enter new positions only if flat
        if position == 0:
            if long_bias and long_breakout and long_pivot and long_volume:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
            elif short_bias and short_breakout and short_pivot and short_volume:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = position * SIGNAL_SIZE
    
    return signals