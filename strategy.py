#!/usr/bin/env python3
"""
exp_6474_1h_donchian20_4h_ema1d_v1
Hypothesis: 1h Donchian(20) breakout with 4h EMA50 trend filter and 1d volume confirmation.
Uses 4h EMA50 to determine bias: long only when price above 4h EMA50,
short only when below. 1d volume MA filters weak breakouts.
Session filter (08-20 UTC) reduces noise trades.
Target: 60-150 total trades over 4 years = 15-37/year for 1h.
Uses discrete position sizing (0.20) to minimize fee churn.
Works in both bull and bear markets by using 4h EMA50 as adaptive trend filter
and Donchian breakouts for momentum entries.
"""
from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_6474_1h_donchian20_4h_ema1d_v1"
timeframe = "1h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOL_MA_PERIOD = 20
VOL_THRESHOLD = 1.5  # volume must be 1.5x its 1d MA
EMA_PERIOD = 50      # 4h EMA for trend filter
SIGNAL_SIZE = 0.20   # 20% position size

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h EMA50
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=EMA_PERIOD, min_periods=EMA_PERIOD, adjust=False).mean().values
    # Align to LTF (1h) with shift(1) for completed bars only
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Calculate 1d volume MA
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=VOL_MA_PERIOD, min_periods=VOL_MA_PERIOD).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    donchian_low = pd.Series(low).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
    # Precompute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, VOL_MA_PERIOD, EMA_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            position = 0
            continue
            
        # Skip if EMA not available (first bar)
        if np.isnan(ema_4h_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]):
            signals[i] = 0.0
            continue
            
        # Long conditions: price breaks above Donchian HIGH + above 4h EMA50 + volume spike
        long_breakout = close[i] > donchian_high[i-1]  # break above previous period's high
        long_bias = close[i] > ema_4h_aligned[i]       # price above 4h EMA50 (bullish bias)
        long_volume = volume[i] > vol_ma_1d_aligned[i] * VOL_THRESHOLD if not np.isnan(vol_ma_1d_aligned[i]) else False
        
        # Short conditions: price breaks below Donchian LOW + below 4h EMA50 + volume spike
        short_breakout = close[i] < donchian_low[i-1]  # break below previous period's low
        short_bias = close[i] < ema_4h_aligned[i]      # price below 4h EMA50 (bearish bias)
        short_volume = volume[i] > vol_ma_1d_aligned[i] * VOL_THRESHOLD if not np.isnan(vol_ma_1d_aligned[i]) else False
        
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