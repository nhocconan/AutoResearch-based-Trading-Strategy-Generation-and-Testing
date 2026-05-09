#!/usr/bin/env python3
# Hypothesis: 4h Camarilla R4/S4 breakout with 1w EMA50 trend filter and volume spike
# Long when price breaks above R4 with 1w EMA50 uptrend and volume > 1.5x average
# Short when price breaks below S4 with 1w EMA50 downtrend and volume > 1.5x average
# Exit when price crosses the central pivot (PP)
# Uses higher timeframe trend filter to reduce noise and false breakouts
# Target: 80-120 total trades over 4 years (20-30/year) with size 0.25

name = "4h_Camarilla_R4S4_Breakout_1wEMA50_VolumeSpike"
timeframe = "4h"
leverage = 1.0

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
    
    # Calculate 1w Camarilla levels (PP, R4, S4)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Previous week's OHLC for Camarilla calculation
    prev_high = df_1w['high'].shift(1)
    prev_low = df_1w['low'].shift(1)
    prev_close = df_1w['close'].shift(1)
    
    # Calculate pivot point
    pp = (prev_high + prev_low + prev_close) / 3
    # Calculate Camarilla R4 and S4 levels (using 1.5 multiplier for outer bands)
    r4 = pp + (prev_high - prev_low) * 1.5000
    s4 = pp - (prev_high - prev_low) * 1.5000
    
    # Align Camarilla levels to 4h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1w, pp.values)
    r4_aligned = align_htf_to_ltf(prices, df_1w, r4.values)
    s4_aligned = align_htf_to_ltf(prices, df_1w, s4.values)
    
    # Calculate 1w EMA50 for trend filter
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume confirmation: current volume > 1.5x 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean()
    vol_confirm = volume > (1.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for EMA calculation
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(pp_aligned[i]) or np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above R4, EMA50 uptrend, volume spike
            if (close[i] > r4_aligned[i] and 
                ema50_1w_aligned[i] > ema50_1w_aligned[i-1] and  # EMA rising
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S4, EMA50 downtrend, volume spike
            elif (close[i] < s4_aligned[i] and 
                  ema50_1w_aligned[i] < ema50_1w_aligned[i-1] and  # EMA falling
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below central pivot
            if close[i] < pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above central pivot
            if close[i] > pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals