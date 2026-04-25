#!/usr/bin/env python3
"""
6h_ElderRay_BullBearPower_1dTrend_VolumeSpike
Hypothesis: 6-hour Elder Ray (Bull/Bear Power) with 1-day EMA50 trend filter and volume spike confirmation.
Targets 12-30 trades/year by requiring: 1) Bull Power > 0 and Bear Power < 0 for strong momentum,
2) aligned with 1d EMA50 trend direction, 3) volume > 2.0x 20-period average for institutional participation.
Uses 6h timeframe to reduce overtrading while capturing sustained moves. Elder Ray measures bull/bear
power relative to EMA13, providing inherent trend strength confirmation. Volume spike ensures moves
are backed by participation, reducing false breakouts. Works in bull markets via strong Bull Power
and in bear markets via strong Bear Power (negative values).
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
    
    # 1d data for EMA50 trend filter (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1d data for EMA13 (used in Elder Ray calculation)
    ema_13_1d = pd.Series(df_1d['close'].values).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema_13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_13_1d)
    
    # Calculate Elder Ray components: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema_13_1d_aligned
    bear_power = low - ema_13_1d_aligned
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for 1d EMA50 (50) and EMA13 (13)
    start_idx = 50 + 13 + 1  # Conservative warmup
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any data not ready
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(vol_ma[i]) or
            np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        
        # Trend filter: price relative to 1d EMA50
        uptrend = curr_close > ema_50_1d_aligned[i]
        downtrend = curr_close < ema_50_1d_aligned[i]
        
        if position == 0:
            # Look for entry signals with Elder Ray confirmation and volume spike
            # Long: Bull Power > 0 (bulls in control) AND Bear Power < 0 (bears weak) AND uptrend AND volume spike
            long_signal = (bull_power[i] > 0) and (bear_power[i] < 0) and uptrend and volume_spike[i]
            # Short: Bear Power < 0 (bears in control) AND Bull Power < 0 (bulls weak) AND downtrend AND volume spike
            short_signal = (bear_power[i] < 0) and (bull_power[i] < 0) and downtrend and volume_spike[i]
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit when Bull Power turns negative (bulls lose control) or trend changes
            if bull_power[i] <= 0 or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit when Bear Power turns positive (bears lose control) or trend changes
            if bear_power[i] >= 0 or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_BullBearPower_1dTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0