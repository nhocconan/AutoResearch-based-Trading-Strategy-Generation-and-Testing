#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_1dTrend_VolumeSpike_v2
Hypothesis: 4h Donchian(20) breakout with daily EMA50 trend filter and volume confirmation.
Targets 20-50 trades/year by requiring: 1) price breaks 4h Donchian(20) channel (strong momentum),
2) aligned with daily EMA50 trend, 3) volume > 2.0x 20-period average. This focuses on capturing
strong trending moves while minimizing false breakouts in choppy markets, suitable for both bull and bear regimes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) once before loop
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 1d data for EMA50 trend filter (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 4h Donchian(20) channels (loaded ONCE)
    donchian_window = 20
    upper_channel = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    lower_channel = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for 1d EMA50 (50) and Donchian(20) (20)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any data not ready
        if (np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        # Trend filter: price relative to 1d EMA50
        uptrend = curr_close > ema_50_1d_aligned[i]
        downtrend = curr_close < ema_50_1d_aligned[i]
        
        if position == 0:
            # Look for entry signals with volume confirmation, trend alignment
            # Long breakout: price breaks above upper Donchian channel with uptrend and volume confirmation
            long_breakout = (curr_high > upper_channel[i]) and uptrend and volume_confirm[i]
            # Short breakout: price breaks below lower Donchian channel with downtrend and volume confirmation
            short_breakout = (curr_low < lower_channel[i]) and downtrend and volume_confirm[i]
            
            if long_breakout:
                signals[i] = 0.30
                position = 1
                entry_price = curr_close
            elif short_breakout:
                signals[i] = -0.30
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit if price breaks below lower channel (mean reversion) or trend changes
            if curr_low < lower_channel[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Short position: exit if price breaks above upper channel (mean reversion) or trend changes
            if curr_high > upper_channel[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

name = "4h_Donchian20_Breakout_1dTrend_VolumeSpike_v2"
timeframe = "4h"
leverage = 1.0