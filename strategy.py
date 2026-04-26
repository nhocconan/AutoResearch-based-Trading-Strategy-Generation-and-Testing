#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_Dyn
Hypothesis: Camarilla pivot levels (R1/S1) from 1d timeframe provide institutional support/resistance. 
Breakout above R1 with 1d uptrend and volume spike (>2x 20-period MA) goes long. 
Breakdown below S1 with 1d downtrend and volume spike goes short.
Uses discrete position sizing (0.25) to minimize fee churn. 
Designed for 12h timeframe to capture multi-day trends while avoiding noise.
Target: 12-37 trades/year (50-150 total over 4 years) with volume confirmation and trend filter reducing false breakouts.
Works in both bull and bear markets by following 1d trend, which adapts to regime changes.
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
    
    # Get 1d data for Camarilla pivot calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivots from previous 1d bar (HLC of completed 1d candle)
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # where C,H,L are from previous completed 1d bar
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Use previous 1d bar's HLC for today's Camarilla levels (no look-ahead)
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d[0] = np.nan  # First bar has no previous
    prev_high_1d[0] = np.nan
    prev_low_1d[0] = np.nan
    
    camarilla_range = prev_high_1d - prev_low_1d
    r1 = prev_close_1d + camarilla_range * 1.1 / 12
    s1 = prev_close_1d - camarilla_range * 1.1 / 12
    
    # Align Camarilla levels to 12h timeframe (available after 1d bar closes)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # 1d EMA34 trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    uptrend_1d = close > ema_34_1d_aligned
    downtrend_1d = close < ema_34_1d_aligned
    
    # Volume confirmation: volume > 2.0x 20-period MA (stricter for fewer trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 34 for 1d EMA + 1 for previous day roll)
    start_idx = 35
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:
            # Long: break above R1 with 1d uptrend and volume spike
            if (close[i] > r1_aligned[i] and 
                uptrend_1d[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below S1 with 1d downtrend and volume spike
            elif (close[i] < s1_aligned[i] and 
                  downtrend_1d[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: 1d trend changes to downtrend OR price breaks below S1 (failed breakout)
            if (not uptrend_1d[i] or close[i] < s1_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: 1d trend changes to uptrend OR price breaks above R1 (failed breakdown)
            if (not downtrend_1d[i] or close[i] > r1_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_Dyn"
timeframe = "12h"
leverage = 1.0