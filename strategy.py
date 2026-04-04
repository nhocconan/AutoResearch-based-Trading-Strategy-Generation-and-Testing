#!/usr/bin/env python3
"""
exp_6554_1h_donchian20_4h_ema1d_vol_v1
Hypothesis: 1h Donchian(20) breakout with 4h EMA50 as trend filter and 1d EMA200 as long-term bias,
plus volume confirmation (2.0x 20-period MA). Uses 4h/1d for signal direction, 1h only for entry timing.
Session filter (08-20 UTC) reduces noise. Discrete sizing (0.20) minimizes fee drag.
Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_6554_1h_donchian20_4h_ema1d_vol_v1"
timeframe = "1h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
EMA_4H_PERIOD = 50      # 4h EMA50 for medium-term trend
EMA_1D_PERIOD = 200     # 1d EMA200 for long-term bias
VOL_MA_PERIOD = 20
VOL_THRESHOLD = 2.0     # volume must be 2.0x its 20-period MA
SIGNAL_SIZE = 0.20      # 20% position size

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Pre-compute session hours (08-20 UTC) - open_time is already datetime64[ms]
    hours = prices.index.hour  # prices.index is DatetimeIndex
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h EMA50
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=EMA_4H_PERIOD, adjust=False).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Calculate 1d EMA200
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=EMA_1D_PERIOD, adjust=False).mean().values
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
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, VOL_MA_PERIOD, EMA_4H_PERIOD, EMA_1D_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(ema_4h_aligned[i]) or np.isnan(ema_1d_aligned[i]):
            continue
            
        # Session filter: only trade 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            position = 0
            continue
        
        # Long conditions: price > 4h EMA50 (bullish bias) AND price > 1d EMA200 (long-term bull) 
        # + breaks above Donchian HIGH + volume spike
        long_bias = close[i] > ema_4h_aligned[i] and close[i] > ema_1d_aligned[i]
        long_breakout = close[i] > donchian_high[i-1]
        long_volume = volume[i] > vol_ma[i] * VOL_THRESHOLD if not np.isnan(vol_ma[i]) else False
        
        # Short conditions: price < 4h EMA50 (bearish bias) AND price < 1d EMA200 (long-term bear)
        # + breaks below Donchian LOW + volume spike
        short_bias = close[i] < ema_4h_aligned[i] and close[i] < ema_1d_aligned[i]
        short_breakout = close[i] < donchian_low[i-1]
        short_volume = volume[i] > vol_ma[i] * VOL_THRESHOLD if not np.isnan(vol_ma[i]) else False
        
        # Exit conditions: EMA reversal or Donchian midpoint reversal
        if position == 1:  # long position
            # Exit if price drops back below 4h EMA50 or 1d EMA200 (trend change)
            exit_long = close[i] < ema_4h_aligned[i] or close[i] < ema_1d_aligned[i]
            # Or if price drops below Donchian midpoint
            exit_long = exit_long or close[i] < (donchian_high[i-1] + donchian_low[i-1]) / 2
            if exit_long:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            # Exit if price rises back above 4h EMA50 or 1d EMA200 (trend change)
            exit_short = close[i] > ema_4h_aligned[i] or close[i] > ema_1d_aligned[i]
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