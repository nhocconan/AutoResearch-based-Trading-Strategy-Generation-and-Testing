#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_1dTrend_VolumeSpike_v1
Hypothesis: 4-hour Donchian(20) breakout with 1-day EMA34 trend filter and volume spike confirmation.
Targets 20-50 trades/year by requiring: 1) price breaks 20-period Donchian channel (strong momentum breakout),
2) aligned with 1d EMA34 trend, 3) volume > 2.0x 20-period average. Uses 4h timeframe to balance
trade frequency and fee drag while capturing significant moves in both bull and bear markets.
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
    
    # Precompute session hours (08-20 UTC) once before loop
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 1d data for EMA34 trend filter (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Donchian(20) channels on 4h data
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for Donchian(20) (20) and 1d EMA34 (34)
    start_idx = 35
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(high_roll[i]) or 
            np.isnan(low_roll[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        # Trend filter: price relative to 1d EMA34
        uptrend = curr_close > ema_34_1d_aligned[i]
        downtrend = curr_close < ema_34_1d_aligned[i]
        
        if position == 0:
            # Look for entry signals with volume confirmation, trend alignment
            # Long breakout: price breaks above upper Donchian with uptrend and volume confirmation
            long_breakout = (curr_close > high_roll[i-1]) and uptrend and volume_confirm[i]
            # Short breakout: price breaks below lower Donchian with downtrend and volume confirmation
            short_breakout = (curr_close < low_roll[i-1]) and downtrend and volume_confirm[i]
            
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
            # Long position: exit conditions
            # Exit if price breaks below lower Donchian (mean reversion) or trend changes
            if curr_close < low_roll[i-1] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Short position: exit conditions
            # Exit if price breaks above upper Donchian (mean reversion) or trend changes
            if curr_close > high_roll[i-1] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

name = "4h_Donchian20_Breakout_1dTrend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0