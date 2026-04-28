#!/usr/bin/env python3
"""
1d_WilliamsAlligator_ElderRay_Vortex_1wTrend
Hypothesis: Combines Williams Alligator (trend), Elder Ray (momentum), and Vortex (direction) on 1d with 1w trend filter.
Williams Alligator identifies trend presence and direction via SMAs. Elder Ray measures bull/bear power via EMA.
Vortex confirms trend direction. 1w trend filter ensures trading with higher timeframe trend.
Designed for low trade frequency (target: 15-30 trades/year) to minimize fee drag.
Works in bull via long signals, in bear via short signals, avoids whipsaw via confluence.
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
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Williams Alligator (13,8,5 SMAs shifted)
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().shift(5).values
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Vortex Indicator (14-period)
    tr1 = np.abs(high - np.roll(low, 1))
    tr2 = np.abs(low - np.roll(high, 1))
    tr = np.maximum(tr1, tr2)
    tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    vm_plus = np.abs(high - np.roll(low, 1))
    vm_minus = np.abs(low - np.roll(high, 1))
    vi_plus = pd.Series(vm_plus).rolling(window=14, min_periods=14).sum().values / tr14
    vi_minus = pd.Series(vm_minus).rolling(window=14, min_periods=14).sum().values / tr14
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(vi_plus[i]) or np.isnan(vi_minus[i])):
            signals[i] = 0.0
            continue
        
        # 1w trend filter
        uptrend_1w = close[i] > ema_50_1w_aligned[i]
        downtrend_1w = close[i] < ema_50_1w_aligned[i]
        
        # Williams Alligator: aligned (jaws < teeth < lips) = uptrend, reversed = downtrend
        alligator_long = (jaw[i] < teeth[i]) and (teeth[i] < lips[i])
        alligator_short = (jaw[i] > teeth[i]) and (teeth[i] > lips[i])
        
        # Elder Ray: bull power > 0 and rising, bear power < 0 and falling
        elder_long = bull_power[i] > 0 and (i == start_idx or bull_power[i] > bull_power[i-1])
        elder_short = bear_power[i] < 0 and (i == start_idx or bear_power[i] < bear_power[i-1])
        
        # Vortex: VI+ > VI- = uptrend, VI- > VI+ = downtrend
        vortex_long = vi_plus[i] > vi_minus[i]
        vortex_short = vi_minus[i] > vi_plus[i]
        
        # Entry logic: confluence of all three indicators in same direction
        long_entry = alligator_long and elder_long and vortex_long and uptrend_1w
        short_entry = alligator_short and elder_short and vortex_short and downtrend_1w
        
        # Exit logic: any indicator fails or 1w trend changes
        long_exit = (not alligator_long) or (not elder_long) or (not vortex_long) or (not uptrend_1w)
        short_exit = (not alligator_short) or (not elder_short) or (not vortex_short) or (not downtrend_1w)
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_WilliamsAlligator_ElderRay_Vortex_1wTrend"
timeframe = "1d"
leverage = 1.0