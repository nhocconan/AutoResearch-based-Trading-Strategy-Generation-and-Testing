#!/usr/bin/env python3
"""
1d_Camarilla_Pivot_R1_S1_Breakout_Volume_Confirmation
Hypothesis: Daily Camarilla pivot levels (R1/S1) act as strong support/resistance.
Breakouts above R1 or below S1 with volume confirmation indicate institutional participation.
Volume filter reduces false breakouts. Works in bull (breakouts continue) and bear (breakdowns continue).
Uses 1w trend filter to avoid counter-trend trades. Target: 15-25 trades/year.
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
    
    # Calculate prior day's Camarilla pivot levels
    # Using prior day's high, low, close
    phigh = np.roll(high, 1)
    plow = np.roll(low, 1)
    pclose = np.roll(close, 1)
    phigh[0] = high[0]
    plow[0] = low[0]
    pclose[0] = close[0]
    
    # Camarilla calculations
    range_val = phigh - plow
    r1 = pclose + range_val * 1.1 / 12
    s1 = pclose - range_val * 1.1 / 12
    
    # Volume confirmation: 20-day average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA20 for trend filter
    close_series_1w = pd.Series(close_1w)
    ema20_1w = close_series_1w.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align 1w EMA to daily timeframe
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 20  # volume MA20 needs 20 periods
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1[i]) or 
            np.isnan(s1[i]) or 
            np.isnan(volume_ma20[i]) or 
            np.isnan(ema20_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-day average
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        # Breakout conditions
        breakout_long = close[i] > r1[i]  # Close above R1
        breakdown_short = close[i] < s1[i]  # Close below S1
        
        if position == 0:
            # Long: breakout above R1 + volume filter + 1w uptrend
            if breakout_long and volume_filter and close[i] > ema20_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: breakdown below S1 + volume filter + 1w downtrend
            elif breakdown_short and volume_filter and close[i] < ema20_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: breakdown below S1 (reversal signal)
            if breakdown_short:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: breakout above R1 (reversal signal)
            if breakout_long:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Camarilla_Pivot_R1_S1_Breakout_Volume_Confirmation"
timeframe = "1d"
leverage = 1.0