#!/usr/bin/env python3
"""
1d_Camarilla_R1_S1_Breakout_1wEMA50_Trend_VolumeSpike_v1
Hypothesis: Daily Camarilla R1/S1 breakout with weekly EMA50 trend filter and volume spike confirmation.
- Uses 1d timeframe for lower trade frequency (target: 30-100 total trades over 4 years)
- Camarilla R1/S1 levels from prior day provide precise support/resistance
- Weekly EMA50 filter ensures alignment with higher timeframe trend (bull/bear agnostic)
- Volume spike (2.5x 20-period average) confirms institutional participation
- Designed for 7-25 trades/year (30-100 total over 4 years) to minimize fee drag
- Works in bull/bear markets by trading with the weekly trend and using volume to filter false breakouts
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate Camarilla levels from previous 1d bar (need 1d data for Camarilla)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_r1 = close_1d + (1.0/12) * (high_1d - low_1d)
    camarilla_s1 = close_1d - (1.0/12) * (high_1d - low_1d)
    
    # Align Camarilla levels to 1d timeframe (use previous day's levels)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Volume spike: volume > 2.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 50 for 1w EMA, 20 for volume MA)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(vol_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Breakout conditions
        breakout_long = close[i] > camarilla_r1_aligned[i]
        breakout_short = close[i] < camarilla_s1_aligned[i]
        
        if position == 0:
            # Long: breakout above R1 AND close > 1w EMA50 AND volume spike
            if breakout_long and close[i] > ema50_1w_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: breakout below S1 AND close < 1w EMA50 AND volume spike
            elif breakout_short and close[i] < ema50_1w_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: breakout below S1
            if breakout_short:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: breakout above R1
            if breakout_long:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Camarilla_R1_S1_Breakout_1wEMA50_Trend_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0