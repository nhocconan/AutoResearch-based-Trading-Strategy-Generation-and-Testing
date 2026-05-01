#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 12h EMA50 trend filter + volume confirmation.
# Long when: price breaks above Donchian(20) high AND 12h EMA50 rising AND volume > 1.5x 20-period average.
# Short when: price breaks below Donchian(20) low AND 12h EMA50 falling AND volume > 1.5x 20-period average.
# Uses discrete sizing 0.25. Target: 20-50 trades/year on 4h.
# Donchian channels provide objective structure, 12h EMA filters counter-trend trades, volume confirms conviction.
# Works in bull (breakouts with trend) and bear (breakdowns with trend) by aligning with higher timeframe direction.

name = "4h_Donchian20_12hEMA50_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours for efficiency
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # Load 12h data ONCE before loop for EMA50 trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 5:
        return np.zeros(n)
    
    # Calculate 12h EMA50
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Rising if current > previous, falling if current < previous
    ema_50_rising = np.roll(ema_50_12h, 1) < ema_50_12h
    ema_50_falling = np.roll(ema_50_12h, 1) > ema_50_12h
    # Handle first value
    ema_50_rising[0] = False
    ema_50_falling[0] = False
    
    # Align 12h EMA50 and trend to 4h
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    ema_50_rising_aligned = align_htf_to_ltf(prices, df_12h, ema_50_rising.astype(float))
    ema_50_falling_aligned = align_htf_to_ltf(prices, df_12h, ema_50_falling.astype(float))
    
    # Donchian(20) on 4h
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for Donchian and EMA
    
    for i in range(start_idx, n):
        # Session filter: 08-20 UTC (reduce noise, focus on active sessions)
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            # Outside session: flatten position if any
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Skip if any data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(ema_50_rising_aligned[i]) or
            np.isnan(ema_50_falling_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_vol_spike = volume_spike[i]
        curr_donchian_high = highest_high[i]
        curr_donchian_low = lowest_low[i]
        curr_ema_50 = ema_50_12h_aligned[i]
        curr_ema_rising = bool(ema_50_rising_aligned[i])
        curr_ema_falling = bool(ema_50_falling_aligned[i])
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: break above Donchian high AND EMA50 rising AND volume spike
            if (curr_close > curr_donchian_high and 
                curr_ema_rising and 
                curr_vol_spike):
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low AND EMA50 falling AND volume spike
            elif (curr_close < curr_donchian_low and 
                  curr_ema_falling and 
                  curr_vol_spike):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price breaks below Donchian low OR EMA50 turns flat/falling
            if (curr_close < curr_donchian_low or 
                not curr_ema_rising):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high OR EMA50 turns flat/rising
            if (curr_close > curr_donchian_high or 
                not curr_ema_falling):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals