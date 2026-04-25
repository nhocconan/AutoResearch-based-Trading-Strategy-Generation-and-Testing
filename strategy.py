#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_1dATR_VolumeSpike
Hypothesis: 4-hour Donchian(20) breakout with 1d ATR volatility filter and volume confirmation.
Targets 20-35 trades/year by requiring: 1) price breaks 20-period Donchian channel, 2) ATR(1d) > 1.5x its 20-period average (high volatility regime), 
3) volume > 1.8x 20-period average. Uses 4h timeframe to capture significant moves while minimizing fee drag.
Volatility filter ensures we trade only during explosive moves, reducing whipsaws in ranging markets.
Works in both bull and bear markets by trading breakouts in direction of the breakout itself (momentum).
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
    
    # 1d data for ATR volatility filter (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    # True Range calculation
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr.rolling(window=14, min_periods=14).mean().values
    atr_ma_20 = pd.Series(atr_1d).rolling(window=20, min_periods=20).mean().values
    volatility_filter = atr_1d > (atr_ma_20 * 1.5)
    
    # Align 1d volatility filter to 4h timeframe
    vol_filter_aligned = align_htf_to_ltf(prices, df_1d, volatility_filter)
    
    # Donchian channel (20-period) on 4h data
    high_ma_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_ma_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.8 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for Donchian (20) + volume MA (20) + 1d ATR (14+20)
    start_idx = 40  # Conservative warmup
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any data not ready
        if (np.isnan(high_ma_20[i]) or np.isnan(low_ma_20[i]) or np.isnan(vol_ma[i]) or
            np.isnan(vol_filter_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        if position == 0:
            # Look for breakout signals with volatility and volume confirmation
            # Long breakout: price breaks above upper Donchian band
            long_breakout = (curr_close > high_ma_20[i]) and vol_filter_aligned[i] and volume_confirm[i]
            # Short breakout: price breaks below lower Donchian band
            short_breakout = (curr_close < low_ma_20[i]) and vol_filter_aligned[i] and volume_confirm[i]
            
            if long_breakout:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_breakout:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit when price retracs to midpoint of channel or volatility drops
            midpoint = (high_ma_20[i] + low_ma_20[i]) / 2.0
            if curr_close < midpoint or not vol_filter_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit when price retracs to midpoint of channel or volatility drops
            midpoint = (high_ma_20[i] + low_ma_20[i]) / 2.0
            if curr_close > midpoint or not vol_filter_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_1dATR_VolumeSpike"
timeframe = "4h"
leverage = 1.0