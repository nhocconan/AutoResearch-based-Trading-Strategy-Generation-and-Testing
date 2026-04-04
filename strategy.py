#!/usr/bin/env python3
"""
exp_6470_1d_donchian20_1w_ema_vol_v1
Hypothesis: Daily Donchian(20) breakout with weekly EMA200 trend filter and volume confirmation.
Uses weekly EMA200 to determine long-term bias: long only when price above weekly EMA200,
short only when below. Volume confirmation filters weak breakouts on daily timeframe.
Designed to work in both bull and bear markets by using weekly EMA200 as adaptive trend filter
and Donchian breakouts for momentum entries. Target: 30-100 trades over 4 years (7-25/year).
"""
from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_6470_1d_donchian20_1w_ema_vol_v1"
timeframe = "1d"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOL_MA_PERIOD = 20
VOL_THRESHOLD = 2.0  # volume must be 2.0x its 20-period MA for confirmation
WEEKLY_EMA_PERIOD = 200  # weekly EMA for long-term trend filter
SIGNAL_SIZE = 0.25   # 25% position size

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    # Calculate weekly EMA200
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=WEEKLY_EMA_PERIOD, min_periods=WEEKLY_EMA_PERIOD, adjust=False).mean().values
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
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, VOL_MA_PERIOD, WEEKLY_EMA_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if weekly EMA not available (first bar)
        if np.isnan(ema_1w_aligned[i]):
            continue
            
        # Long conditions: price breaks above Donchian HIGH + above weekly EMA200 + volume spike
        long_breakout = close[i] > donchian_high[i-1]  # break above previous period's high
        long_bias = close[i] > ema_1w_aligned[i]       # price above weekly EMA200 (bullish bias)
        long_volume = volume[i] > vol_ma[i] * VOL_THRESHOLD if not np.isnan(vol_ma[i]) else False
        
        # Short conditions: price breaks below Donchian LOW + below weekly EMA200 + volume spike
        short_breakout = close[i] < donchian_low[i-1]  # break below previous period's low
        short_bias = close[i] < ema_1w_aligned[i]      # price below weekly EMA200 (bearish bias)
        short_volume = volume[i] > vol_ma[i] * VOL_THRESHOLD if not np.isnan(vol_ma[i]) else False
        
        # Exit conditions: ATR-based stoploss approximation
        # Simple approach: exit when price reverses halfway through the channel
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