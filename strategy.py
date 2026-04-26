#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_VolumeSpike_Regime_v1
Hypothesis: Camarilla pivot R1/S1 breakout with volume spike and 1d trend regime.
Long on R1 breakout in 1d uptrend; short on S1 breakout in 1d downtrend.
Volume confirmation reduces false breakouts. Discrete sizing (0.25) limits fee churn.
Target: 75-200 trades over 4 years = 19-50/year. Works in bull (trend continuation) and bear (counter-trend bounces) via regime filter.
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
    
    # Get 1d data for Camarilla pivots and regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d bar
    # R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    rang = prev_high - prev_low
    R1 = prev_close + 1.1 * rang / 12
    S1 = prev_close - 1.1 * rang / 12
    
    # Align to 4h - note: already delayed by shift(1) for previous day
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # 1d EMA(34) for regime filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need at least 1d data + volume MA
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(R1_aligned[i]) or
            np.isnan(S1_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        close_val = close[i]
        r1 = R1_aligned[i]
        s1 = S1_aligned[i]
        vol_conf = volume_confirm[i]
        regime_long = close_val > ema_34_1d_aligned[i]  # 1d uptrend
        regime_short = close_val < ema_34_1d_aligned[i]  # 1d downtrend
        
        if position == 0:
            # Long: close breaks above R1 AND volume confirm AND 1d uptrend
            long_signal = (close_val > r1) and vol_conf and regime_long
            
            # Short: close breaks below S1 AND volume confirm AND 1d downtrend
            short_signal = (close_val < s1) and vol_conf and regime_short
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: close crosses below S1 (reversal) OR 1d trend flips down
            if (close_val < s1) or (not regime_long):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: close crosses above R1 (reversal) OR 1d trend flips up
            if (close_val > r1) or (not regime_short):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_VolumeSpike_Regime_v1"
timeframe = "4h"
leverage = 1.0