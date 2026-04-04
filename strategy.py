#!/usr/bin/env python3
"""
exp_6578_1d_donchian20_1w_ema_vol_v1
Hypothesis: 1d Donchian(20) breakout with 1w EMA trend filter and volume confirmation.
Uses 1d primary timeframe to minimize fee drag (target: 30-100 total trades over 4 years).
1w EMA provides major trend direction to filter breakouts, working in both bull and bear markets:
- In bull markets: only take longs above 1w EMA
- In bear markets: only take shorts below 1w EMA
Volume confirmation ensures breakouts have conviction. Discrete sizing (0.25) minimizes fee churn.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_6578_1d_donchian20_1w_ema_vol_v1"
timeframe = "1d"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
EMA_PERIOD = 50  # 1w EMA (approx 50 trading days)
VOL_MA_PERIOD = 20
VOL_BASE_THRESHOLD = 2.0  # Volume threshold for confirmation
SIGNAL_SIZE = 0.25      # 25% position size
MAX_HOLD_BARS = 60      # Max hold: ~60 days (1d bars)

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 1w for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w EMA for trend filter
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=EMA_PERIOD, min_periods=EMA_PERIOD, adjust=False).mean().values
    
    # Align to LTF (1d) with shift(1) for completed bars only
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
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
    bars_since_entry = 0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, VOL_MA_PERIOD, EMA_PERIOD) + 1
    
    for i in range(start, n):
        bars_since_entry += 1
        
        # Skip if HTF data not available
        if (np.isnan(ema_1w_aligned[i]) or 
            np.isnan(vol_ma[i])):
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
        
        # Enter new positions only if flat
        if position == 0:
            # Long conditions: 
            # 1. Break above Donchian HIGH (breakout)
            # 2. Volume confirmation
            # 3. Price above 1w EMA (bullish trend filter)
            long_breakout = close[i] > donchian_high[i-1]
            long_volume = volume[i] > vol_ma[i] * VOL_BASE_THRESHOLD if not np.isnan(vol_ma[i]) else False
            long_trend = close[i] > ema_1w_aligned[i]
            
            # Short conditions:
            # 1. Break below Donchian LOW (breakdown)
            # 2. Volume confirmation
            # 3. Price below 1w EMA (bearish trend filter)
            short_breakout = close[i] < donchian_low[i-1]
            short_volume = volume[i] > vol_ma[i] * VOL_BASE_THRESHOLD if not np.isnan(vol_ma[i]) else False
            short_trend = close[i] < ema_1w_aligned[i]
            
            if long_breakout and long_volume and long_trend:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                bars_since_entry = 0
            elif short_breakout and short_volume and short_trend:
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