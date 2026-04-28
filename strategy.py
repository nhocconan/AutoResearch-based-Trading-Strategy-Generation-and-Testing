#!/usr/bin/env python3
"""
4h_Vortex_Volume_Trend_Breakout_v1
Hypothesis: Vortex indicator (VI+ and VI-) identifies trend direction, combined with volume surge and price breakout above/below prior session high/low. 
This captures strong trending moves with institutional participation, avoiding chop. Works in bull (trend continuation) and bear (sharp reversals) by filtering for volume confirmation.
Target: 20-30 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend context and volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Daily ATR for volatility filter (avoid low-vol chop)
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr.rolling(window=14, min_periods=14).mean().values
    # Normalize ATR by price to get % volatility
    atr_pct = atr_1d / df_1d['close'].values
    atr_pct_ma = pd.Series(atr_pct).rolling(window=30, min_periods=30).mean().values
    low_vol_filter = atr_pct > (atr_pct_ma * 0.8)  # Avoid extremely low vol
    
    # Vortex Indicator on 4h data (requires true range and directional movement)
    # VM+ = |high - low_prev|, VM- = |low - high_prev|
    vm_plus = abs(high - np.roll(low, 1))
    vm_minus = abs(low - np.roll(high, 1))
    # First value is invalid due to roll, set to 0
    vm_plus[0] = 0
    vm_minus[0] = 0
    # True range
    tr1 = high - low
    tr2 = abs(high - np.roll(close, 1))
    tr3 = abs(low - np.roll(close, 1))
    tr_4h = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_4h[0] = tr1[0]  # First TR is just high-low
    # Smooth over 14 periods
    vm_plus_sum = pd.Series(vm_plus).rolling(window=14, min_periods=14).sum().values
    vm_minus_sum = pd.Series(vm_minus).rolling(window=14, min_periods=14).sum().values
    tr_sum = pd.Series(tr_4h).rolling(window=14, min_periods=14).sum().values
    vi_plus = vm_plus_sum / tr_sum
    vi_minus = vm_minus_sum / tr_sum
    
    # Trend: VI+ > VI- indicates uptrend, VI- > VI+ indicates downtrend
    vi_uptrend = vi_plus > vi_minus
    vi_downtrend = vi_minus > vi_plus
    
    # Breakout: price > prior 24-period high (for long) or < prior 24-period low (for short)
    # Using 24 periods = 4 days of 4h data for session breakout
    high_24 = pd.Series(high).rolling(window=24, min_periods=24).max().values
    low_24 = pd.Series(low).rolling(window=24, min_periods=24).min().values
    breakout_up = close > high_24
    breakout_down = close < low_24
    
    # Volume confirmation: current volume > 1.8x 24-period average
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_surge = volume > (vol_ma_24 * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(vi_plus[i]) or np.isnan(vi_minus[i]) or 
            np.isnan(high_24[i]) or np.isnan(low_24[i]) or 
            np.isnan(volume_surge[i]) or np.isnan(low_vol_filter[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions: breakout + trend alignment + volume surge + vol filter
        long_entry = breakout_up[i] and vi_uptrend[i] and volume_surge[i] and low_vol_filter[i]
        short_entry = breakout_down[i] and vi_downtrend[i] and volume_surge[i] and low_vol_filter[i]
        
        # Exit when trend reverses (VI crosses) OR opposite breakout with volume
        trend_exit_long = vi_downtrend[i]  # VI- crosses above VI+
        trend_exit_short = vi_uptrend[i]   # VI+ crosses above VI-
        opposite_breakout = (breakout_down[i] and volume_surge[i]) or (breakout_up[i] and volume_surge[i])
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif (trend_exit_long or opposite_breakout) and position == 1:
            signals[i] = -0.25  # Reverse to short
            position = -1
        elif (trend_exit_short or opposite_breakout) and position == -1:
            signals[i] = 0.25   # Reverse to long
            position = 1
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_Vortex_Volume_Trend_Breakout_v1"
timeframe = "4h"
leverage = 1.0