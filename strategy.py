#!/usr/bin/env python3
name = "4h_Camarilla_R1_S1_Breakout_1dEMA34_VolumeSpike_Dyn_v2"
timeframe = "4h"
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
    
    # ===== 1d Trend Filter (HTF) =====
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1d EMA(34) for trend
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # ===== 1d Camarilla Pivot Levels (HTF) =====
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_prev = np.roll(close_1d, 1)
    close_1d_prev[0] = close_1d[0]  # first day uses same close
    
    pivot = (high_1d + low_1d + close_1d_prev) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla levels: R1 = close + (high - low) * 1.1/12, S1 = close - (high - low) * 1.1/12
    r1 = close_1d_prev + range_1d * 1.1 / 12
    s1 = close_1d_prev - range_1d * 1.1 / 12
    
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # ===== Volume Spike Filter =====
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_avg)
    
    # ===== Additional Filter: Minimum Holding Period (4 bars = 16h) =====
    # Prevents whipsaw and reduces trade frequency
    hold_timer = np.zeros(n, dtype=int)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                hold_timer[i] = 0
            else:
                signals[i] = 0.0
                hold_timer[i] = 0
            continue
        
        # Decrease hold timer if in position
        if position != 0:
            hold_timer[i] = hold_timer[i-1] - 1 if i > 0 else 0
        else:
            hold_timer[i] = 0
        
        if position == 0:
            # Only allow new entry if hold timer expired (should be 0 when flat)
            if hold_timer[i] <= 0:
                # Long: Close crosses above R1 + above 1d EMA34 + volume spike
                if (close[i] > r1_aligned[i] and close[i-1] <= r1_aligned[i-1] and
                    close[i] > ema34_1d_aligned[i] and
                    vol_spike[i]):
                    signals[i] = 0.25
                    position = 1
                    hold_timer[i] = 4  # Hold for 4 bars minimum
                # Short: Close crosses below S1 + below 1d EMA34 + volume spike
                elif (close[i] < s1_aligned[i] and close[i-1] >= s1_aligned[i-1] and
                      close[i] < ema34_1d_aligned[i] and
                      vol_spike[i]):
                    signals[i] = -0.25
                    position = -1
                    hold_timer[i] = 4  # Hold for 4 bars minimum
        elif position == 1:
            # Exit conditions: Close crosses below S1 OR below 1d EMA34
            # But only after minimum holding period
            if hold_timer[i] <= 0:
                if close[i] < s1_aligned[i] or close[i] < ema34_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:
                # Still in minimum hold period, maintain position
                signals[i] = 0.25
        elif position == -1:
            # Exit conditions: Close crosses above R1 OR above 1d EMA34
            # But only after minimum holding period
            if hold_timer[i] <= 0:
                if close[i] > r1_aligned[i] or close[i] > ema34_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:
                # Still in minimum hold period, maintain position
                signals[i] = -0.25
    
    return signals