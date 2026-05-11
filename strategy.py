#!/usr/bin/env python3
name = "12h_Camarilla_R1S1_Breakout_1dTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0

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
    
    # 1d EMA34 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Daily Camarilla pivot levels (R1, S1)
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    pivot = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    R1 = close_1d + (range_1d * 1.1 / 12)
    S1 = close_1d - (range_1d * 1.1 / 12)
    
    # Align to 12h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # Volume filter: 20-period EMA on 12h timeframe
    vol_ema20 = pd.Series(volume).ewm(span=20, min_periods=20, adjust=False).mean().values
    volume_ok = volume > vol_ema20 * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(pivot_aligned[i]) or 
            np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or 
            np.isnan(volume_ok[i])):
            if position == 1:
                signals[i] = 0.30
            elif position == -1:
                signals[i] = -0.30
            else:
                signals[i] = 0.0
            continue
        
        # Conditions
        price_above_ema1d = close[i] > ema34_1d_aligned[i]
        price_below_ema1d = close[i] < ema34_1d_aligned[i]
        breakout_long = close[i] > R1_aligned[i]
        breakout_short = close[i] < S1_aligned[i]
        
        if position == 0:
            # Long: Price breaks above R1 + above 1d EMA34 + volume spike
            if breakout_long and price_above_ema1d and volume_ok[i]:
                signals[i] = 0.30
                position = 1
            # Short: Price breaks below S1 + below 1d EMA34 + volume spike
            elif breakout_short and price_below_ema1d and volume_ok[i]:
                signals[i] = -0.30
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit: Price crosses below pivot OR trend reverses
                if close[i] < pivot_aligned[i] or close[i] < ema34_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.30
            elif position == -1:
                # Exit: Price crosses above pivot OR trend reverses
                if close[i] > pivot_aligned[i] or close[i] > ema34_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.30
    
    return signals