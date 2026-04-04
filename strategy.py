#!/usr/bin/env python3
"""
exp_6554_1h_donchian20_4h_ema1d_vol_v1
Hypothesis: 1h Donchian(20) breakout with 4h EMA200 as trend filter and 1d volume confirmation.
Uses 4h EMA200 for medium-term trend identification and 1d volume spike (2.0x 20-period MA) for confirmation.
Designed for 60-150 total trades over 4 years (15-37/year) with discrete sizing (0.20) to minimize fee drag.
Session filter: 08-20 UTC to avoid low-liquidity hours.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_6554_1h_donchian20_4h_ema1d_vol_v1"
timeframe = "1h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
EMA_PERIOD_4H = 200     # 4h EMA200 for medium-term trend filter
VOL_MA_PERIOD_1D = 20
VOL_THRESHOLD = 2.0     # volume must be 2.0x its 20-period MA
SIGNAL_SIZE = 0.20      # 20% position size

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')   # for EMA200
    df_1d = get_htf_data(prices, '1d')   # for volume MA
    
    # Calculate 4h EMA200
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=EMA_PERIOD_4H, adjust=False).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Calculate 1d volume MA
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=VOL_MA_PERIOD_1D, min_periods=VOL_MA_PERIOD_1D).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    donchian_low = pd.Series(low).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, EMA_PERIOD_4H, VOL_MA_PERIOD_1D) + 1
    
    for i in range(start, n):
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            position = 0
            continue
            
        # Skip if HTF data not available
        if np.isnan(ema_4h_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]):
            continue
            
        # Long conditions: price > 4h EMA200 (bullish bias) + breaks above Donchian HIGH + 1d volume spike
        long_bias = close[i] > ema_4h_aligned[i]  # price above 4h EMA200 (bullish)
        long_breakout = close[i] > donchian_high[i-1]  # break above previous period's high
        long_volume = volume[i] > vol_ma_1d_aligned[i] * VOL_THRESHOLD
        
        # Short conditions: price < 4h EMA200 (bearish bias) + breaks below Donchian LOW + 1d volume spike
        short_bias = close[i] < ema_4h_aligned[i]  # price below 4h EMA200 (bearish)
        short_breakout = close[i] < donchian_low[i-1]  # break below previous period's low
        short_volume = volume[i] > vol_ma_1d_aligned[i] * VOL_THRESHOLD
        
        # Exit conditions: EMA reversal or Donchian midpoint reversal
        if position == 1:  # long position
            # Exit if price drops back below EMA200 (trend change)
            exit_long = close[i] < ema_4h_aligned[i]
            # Or if price drops below Donchian midpoint
            exit_long = exit_long or close[i] < (donchian_high[i-1] + donchian_low[i-1]) / 2
            if exit_long:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            # Exit if price rises back above EMA200 (trend change)
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