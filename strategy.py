#!/usr/bin/env python3
"""
12h_Camarilla_R1S1_Breakout_1dTrend_Volume
Hypothesis: Camarilla R1/S1 breakouts on 12h with 1d EMA34 trend filter and volume confirmation.
Works in bull/bear via trend filter. Target: 12-37 trades/year on 12h to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_price = prices['open'].values
    
    # Get 1d data for trend and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 1d trend: EMA34
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate Camarilla levels from previous 1d bar (HLC of previous day)
    ph = df_1d['high'].values  # previous day high
    pl = df_1d['low'].values   # previous day low
    pc = df_1d['close'].values # previous day close
    
    # Camarilla R1 and S1
    r1 = pc + (ph - pl) * 1.1 / 12
    s1 = pc - (ph - pl) * 1.1 / 12
    
    # Align Camarilla levels to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume confirmation: volume > 2.0 * 24-period average (24*12h = 12 days)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for EMA and volume MA
    start_idx = max(34, 24)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if np.isnan(ema34_1d_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]):
            signals[i] = 0.0
            continue
        
        ema_trend = ema34_1d_aligned[i]
        r1_level = r1_aligned[i]
        s1_level = s1_aligned[i]
        vol_spike_val = vol_spike[i]
        
        if position == 0:
            # Long: price breaks above R1 + uptrend + volume spike
            if close[i] > r1_level and open_price[i] <= r1_level and close[i] > ema_trend and vol_spike_val:
                signals[i] = size
                position = 1
            # Short: price breaks below S1 + downtrend + volume spike
            elif close[i] < s1_level and open_price[i] >= s1_level and close[i] < ema_trend and vol_spike_val:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below S1 or trend turns down
            if close[i] < s1_level or close[i] < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price breaks above R1 or trend turns up
            if close[i] > r1_level or close[i] > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0