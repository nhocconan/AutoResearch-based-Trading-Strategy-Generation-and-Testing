#!/usr/bin/env python3
"""
exp_6574_1h_donchian20_1d_ema_vol_v1
Hypothesis: 1h Donchian(20) breakout with 1d EMA(50) direction filter and volume confirmation.
Uses 1h primary timeframe with 1d HTF for trend filter to balance trade frequency (target: 60-150 total trades over 4 years).
1d EMA(50) provides institutional trend bias that works in both bull and bear markets:
- Above EMA50: bullish bias (favor breakout longs)
- Below EMA50: bearish bias (favor breakdown shorts)
Volume confirmation ensures breakouts have conviction. Discrete sizing (0.20) minimizes fee churn.
Session filter (08-20 UTC) reduces noise trades.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_6574_1h_donchian20_1d_ema_vol_v1"
timeframe = "1h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
EMA_PERIOD = 50
VOL_MA_PERIOD = 20
VOL_BASE_THRESHOLD = 2.0  # Volume threshold for confirmation
SIGNAL_SIZE = 0.20      # 20% position size
MAX_HOLD_BARS = 24      # Max hold: 24 hours (1h bars)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 1d for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA(50) for trend bias
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=EMA_PERIOD, min_periods=EMA_PERIOD, adjust=False).mean().values
    
    # Align to LTF (1h) with shift(1) for completed bars only
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
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
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, VOL_MA_PERIOD, EMA_PERIOD) + 1
    
    for i in range(start, n):
        bars_since_entry += 1
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            # Outside session: flatten position
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.0
            continue
        
        # Skip if HTF data not available
        if np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
        
        # Exit conditions: time-based exit OR Donchian midpoint reversal
        if position == 1:  # long position
            # Exit if price drops below Donchian midpoint
            exit_long = close[i] < (donchian_high[i-1] + donchian_low[i-1]) / 2
            # Time-based exit: prevent overstaying
            exit_long = exit_long or bars_since_entry >= MAX_HOLD_BARS
            if exit_long:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
                continue
        elif position == -1:  # short position
            # Exit if price rises above Donchian midpoint
            exit_short = close[i] > (donchian_high[i-1] + donchian_low[i-1]) / 2
            # Time-based exit: prevent overstaying
            exit_short = exit_short or bars_since_entry >= MAX_HOLD_BARS
            if exit_short:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
                continue
        
        # Determine trend bias from 1d EMA
        bullish_bias = close[i] > ema_1d_aligned[i]
        bearish_bias = close[i] < ema_1d_aligned[i]
        
        # Long conditions: 
        # 1. Break above Donchian HIGH (breakout)
        # 2. Volume confirmation
        # 3. Bullish bias from 1d EMA
        long_breakout = close[i] > donchian_high[i-1]
        long_volume = volume[i] > vol_ma[i] * VOL_BASE_THRESHOLD if not np.isnan(vol_ma[i]) else False
        
        # Short conditions:
        # 1. Break below Donchian LOW (breakdown)
        # 2. Volume confirmation
        # 3. Bearish bias from 1d EMA
        short_breakout = close[i] < donchian_low[i-1]
        short_volume = volume[i] > vol_ma[i] * VOL_BASE_THRESHOLD if not np.isnan(vol_ma[i]) else False
        
        # Enter new positions only if flat
        if position == 0:
            if long_breakout and long_volume and bullish_bias:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                bars_since_entry = 0
            elif short_breakout and short_volume and bearish_bias:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                bars_since_entry = 0
            else:
                signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = position * SIGNAL_SIZE
    
    return signals