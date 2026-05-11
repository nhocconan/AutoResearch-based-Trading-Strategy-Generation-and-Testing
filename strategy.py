#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike
Hypothesis: Uses daily Camarilla pivot levels (R1, S1) with 1-day EMA34 trend filter and volume spike confirmation.
Works in bull markets via breakouts above R1 and bear markets via breakdowns below S1. Volume spike filters false breakouts.
Target: 15-25 trades/year to minimize fee drag while capturing strong directional moves.
"""

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for Camarilla pivots and EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 12h OHLCV
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # --- 1d Camarilla Pivot Levels (R1, S1) ---
    # Calculate pivot point and ranges
    pp = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    range_hl = df_1d['high'] - df_1d['low']
    
    # Camarilla R1 and S1
    r1 = pp + (range_hl * 1.1 / 12)
    s1 = pp - (range_hl * 1.1 / 12)
    
    # Align Camarilla levels to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1.values)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1.values)
    
    # --- 1d EMA34 for trend filter ---
    ema_34 = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean()
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34.values)
    
    # --- Volume Spike Detection (20-period EMA) ---
    vol_ma = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean()
    vol_spike = volume > (2.0 * vol_ma.values)  # Significant volume spike
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or
            np.isnan(ema_34_aligned[i]) or
            np.isnan(vol_spike[i])):
            # Maintain position if valid, otherwise flat
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Determine market trend based on price vs EMA34
        is_uptrend = close[i] > ema_34_aligned[i]
        is_downtrend = close[i] < ema_34_aligned[i]
        
        # Breakout signals with volume confirmation
        long_breakout = (high[i] > r1_aligned[i]) and vol_spike[i]
        short_breakout = (low[i] < s1_aligned[i]) and vol_spike[i]
        
        if position == 0:
            # Only take long breaks in uptrend, short breaks in downtrend
            if is_uptrend and long_breakout:
                signals[i] = 0.25
                position = 1
            elif is_downtrend and short_breakout:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit long: price touches S1 (opposite level) or trend changes
                exit_signal = (low[i] < s1_aligned[i]) or (not is_uptrend)
                if exit_signal:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price touches R1 (opposite level) or trend changes
                exit_signal = (high[i] > r1_aligned[i]) or (not is_downtrend)
                if exit_signal:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals