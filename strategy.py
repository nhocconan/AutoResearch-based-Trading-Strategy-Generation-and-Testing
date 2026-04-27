#!/usr/bin/env python3
"""
12h_Camarilla_Pivot_R1_S1_Breakout_1dTrend_Volume
Hypothesis: Camarilla pivot breakouts with 1d EMA34 trend and volume confirmation yield high-probability momentum trades.
Works in bull/bear via trend filter. Targets 20-40 trades/year on 12h to minimize fee drag while capturing strong moves.
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
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot and levels using previous day's data
    pivot = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    r1 = pivot + (range_1d * 1.1 / 12)  # Resistance 1
    s1 = pivot - (range_1d * 1.1 / 12)  # Support 1
    
    # Align to 12h timeframe (wait for 1d bar to close)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    
    # Get 1d EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for EMA and volume MA
    start_idx = max(35, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(ema34_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        ema_trend = ema34_1d_aligned[i]
        vol_confirm_val = vol_confirm[i]
        
        if position == 0:
            # Long: break above R1 with uptrend and volume
            if close[i] > r1_val and vol_confirm_val and close[i] > ema_trend:
                signals[i] = size
                position = 1
            # Short: break below S1 with downtrend and volume
            elif close[i] < s1_val and vol_confirm_val and close[i] < ema_trend:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: break below pivot or trend turns down
            if close[i] < pivot_aligned[i] or close[i] < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: break above pivot or trend turns up
            if close[i] > pivot_aligned[i] or close[i] > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Camarilla_Pivot_R1_S1_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0