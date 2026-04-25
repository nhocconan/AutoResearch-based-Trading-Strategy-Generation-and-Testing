#!/usr/bin/env python3
"""
12h Donchian(20) Breakout + Daily EMA50 Trend + Volume Spike
Hypothesis: Donchian channel breakouts on 12h capture medium-term trends, filtered by daily EMA50 trend alignment and volume spikes to avoid false breakouts. Works in bull markets (trend continuation) and bear markets (failed breaks reverse to opposite band). 12h timeframe targets 12-37 trades/year to minimize fee drag while allowing sufficient samples.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data for EMA50 trend filter (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    
    # Daily EMA50 trend
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for daily EMA and volume MA
    start_idx = max(50, 20) + 5
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        # Calculate Donchian bands from last 20 periods (including current)
        lookback_start = max(0, i - 19)
        period_high = np.max(high[lookback_start:i+1])
        period_low = np.min(low[lookback_start:i+1])
        
        curr_close = close[i]
        vol_spike = volume_spike[i]
        
        # Breakout conditions
        breakout_long = curr_close > period_high
        breakout_short = curr_close < period_low
        
        if position == 0:
            # Look for entry signals - require: Donchian breakout + volume spike + daily EMA50 trend alignment
            long_entry = breakout_long and vol_spike and (curr_close > ema_50_1d_aligned[i])
            short_entry = breakout_short and vol_spike and (curr_close < ema_50_1d_aligned[i])
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit on close below Donchian low or trend change
            if curr_close < period_low or curr_close < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit on close above Donchian high or trend change
            if curr_close > period_high or curr_close > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_Breakout_1dEMA50_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0