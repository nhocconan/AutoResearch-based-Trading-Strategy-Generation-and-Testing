#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_Volume_Spike
Hypothesis: Camarilla pivot levels from daily timeframe provide high-probability support/resistance.
Breakouts above R1 or below S1 with 1d EMA50 trend filter and volume spike confirmation.
Designed for 12-37 trades per year on 12h timeframe, works in bull via breakouts above R1 with EMA50 uptrend,
and bear via breakdowns below S1 with EMA50 downtrend. Uses tight entry conditions to minimize fee drag.
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
    
    # Get 1d data for Camarilla pivots and EMA filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's values (shift by 1)
    phigh = np.concatenate([[np.nan], high_1d[:-1]])
    plow = np.concatenate([[np.nan], low_1d[:-1]])
    pclose = np.concatenate([[np.nan], close_1d[:-1]])
    
    # Camarilla R1 and S1
    camarilla_range = phigh - plow
    r1 = pclose + camarilla_range * 1.1 / 12
    s1 = pclose - camarilla_range * 1.1 / 12
    
    # Align to 12h timeframe (wait for 1d close)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume spike: volume > 2.0 * 20-period average (higher threshold for lower frequency)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for calculations
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        ema_50_val = ema_50_1d_aligned[i]
        vol_spike_val = vol_spike[i]
        
        if position == 0:
            # Long: break above R1 AND price above EMA50 AND volume spike
            if close[i] > r1_val and close[i] > ema_50_val and vol_spike_val:
                signals[i] = size
                position = 1
            # Short: break below S1 AND price below EMA50 AND volume spike
            elif close[i] < s1_val and close[i] < ema_50_val and vol_spike_val:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: price drops below S1 (reversal) OR trend change (price below EMA50)
            if close[i] < s1_val or close[i] < ema_50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price rises above R1 (reversal) OR trend change (price above EMA50)
            if close[i] > r1_val or close[i] > ema_50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_Volume_Spike"
timeframe = "12h"
leverage = 1.0