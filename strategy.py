#!/usr/bin/env python3
"""
exp_6534_1h_donchian20_4h_ema_vol_v1
Hypothesis: 1h Donchian(20) breakout with 4h EMA50 as trend filter and volume confirmation.
Uses 1h for entry timing, 4h for signal direction to reduce trade frequency.
Volume spike (2.0x) confirms breakout strength.
Session filter (08-20 UTC) reduces noise trades.
Target: 60-150 total trades over 4 years = 15-37/year for 1h timeframe.
Position size: 0.20 (20%).
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_6534_1h_donchian20_4h_ema_vol_v1"
timeframe = "1h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
EMA_PERIOD = 50
VOL_MA_PERIOD = 20
VOL_THRESHOLD = 2.0
SIGNAL_SIZE = 0.20  # 20% position size

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Pre-compute session hours ONCE (avoid datetime64 arithmetic in loop)
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    
    # Load HTF data ONCE before loop - using 4h for EMA50
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h EMA50
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=EMA_PERIOD, adjust=False).mean().values
    
    # Align to LTF (1h) with shift(1) for completed bars only
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
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
        # Session filter: 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
            
        # Skip if HTF data not available
        if np.isnan(ema_4h_aligned[i]):
            signals[i] = 0.0
            continue
            
        # Long conditions: price > 4h EMA50 (bullish bias) + breaks above Donchian HIGH + volume spike
        long_bias = close[i] > ema_4h_aligned[i]  # price above 4h EMA50 (bullish)
        long_breakout = close[i] > donchian_high[i-1]  # break above previous period's high
        long_volume = volume[i] > vol_ma[i] * VOL_THRESHOLD if not np.isnan(vol_ma[i]) else False
        
        # Short conditions: price < 4h EMA50 (bearish bias) + breaks below Donchian LOW + volume spike
        short_bias = close[i] < ema_4h_aligned[i]  # price below 4h EMA50 (bearish)
        short_breakout = close[i] < donchian_low[i-1]  # break below previous period's low
        short_volume = volume[i] > vol_ma[i] * VOL_THRESHOLD if not np.isnan(vol_ma[i]) else False
        
        # Exit conditions: EMA reversal or Donchian midpoint reversal
        if position == 1:  # long position
            # Exit if price drops back below 4h EMA50 (trend change)
            exit_long = close[i] < ema_4h_aligned[i]
            # Or if price drops below Donchian midpoint
            exit_long = exit_long or close[i] < (donchian_high[i-1] + donchian_low[i-1]) / 2
            if exit_long:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            # Exit if price rises back above 4h EMA50 (trend change)
            exit_short = close[i] > ema_4h_aligned[i]
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