#!/usr/bin/env python3
"""
4h Camarilla R1/S1 Breakout with 1d EMA34 Trend and Volume Spike
Hypothesis: Camarilla pivot levels (R1/S1) from daily charts act as key intraday support/resistance.
Breakouts above R1 or below S1 with volume confirmation and aligned with 1d EMA34 trend capture
strong momentum moves. Works in both bull (breakouts with trend) and bear (breakdowns with trend)
markets by trading in direction of higher timeframe trend. Target: 20-50 trades/year on 4h.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla pivot levels from previous day
    # Need previous day's high, low, close - use 1d data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC (already completed 1d bar due to get_htf_data)
    prev_high = df_1d['high'].shift(1).values  # shift(1) for previous completed day
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Calculate Camarilla levels
    camarilla_range = prev_high - prev_low
    r1 = prev_close + camarilla_range * 1.1 / 12
    s1 = prev_close - camarilla_range * 1.1 / 12
    
    # Align to 4h timeframe (wait for 1d bar to close)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    # 1d EMA34 trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for all indicators
    start_idx = max(20, 34) + 2  # extra buffer for 1d alignment
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(vol_ma[i]) or np.isnan(ema_34_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        
        if position == 0:
            # Look for entry signals - require: Camarilla breakout + volume spike + 1d EMA trend alignment
            long_entry = (curr_close > r1_aligned[i]) and volume_spike[i] and (curr_close > ema_34_1d_aligned[i])
            short_entry = (curr_close < s1_aligned[i]) and volume_spike[i] and (curr_close < ema_34_1d_aligned[i])
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit on close below S1 (mean reversion) or reverse signal
            if curr_close < s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit on close above R1 (mean reversion) or reverse signal
            if curr_close > r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_1dEMA34_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0