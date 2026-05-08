# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
12h_1d_Camarilla_Pivot_Breakout_With_Volume
Timeframe: 12h
Hypothesis: Camarilla pivot levels from daily timeframe act as strong support/resistance.
Breakouts above R3 or below S3 with volume confirmation and aligned weekly trend capture institutional flow.
Works in both bull and bear markets by filtering breakouts with higher-timeframe trend and volume spikes.
Target: 12-30 trades/year (~50-120 over 4 years) to avoid fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_Camarilla_Pivot_Breakout_With_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    # R4 = C + ((H-L) * 1.5000), R3 = C + ((H-L) * 1.2500), etc.
    # S4 = C - ((H-L) * 1.5000), S3 = C - ((H-L) * 1.2500), etc.
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Calculate pivot levels
    R3 = prev_close + (prev_high - prev_low) * 1.2500
    S3 = prev_close - (prev_high - prev_low) * 1.2500
    R4 = prev_close + (prev_high - prev_low) * 1.5000
    S4 = prev_close - (prev_high - prev_low) * 1.5000
    
    # Align daily levels to 12h timeframe
    R3_12h = align_htf_to_ltf(prices, df_1d, R3)
    S3_12h = align_htf_to_ltf(prices, df_1d, S3)
    R4_12h = align_htf_to_ltf(prices, df_1d, R4)
    S4_12h = align_htf_to_ltf(prices, df_1d, S4)
    
    # Weekly EMA34 for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume spike: current volume > 2.0x 24-period average (2 days of 12h data)
    vol_ma24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (2.0 * vol_ma24)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(R3_12h[i]) or np.isnan(S3_12h[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above R3 with volume spike and weekly uptrend
            long_cond = (close[i] > R3_12h[i] and 
                        volume_spike[i] and
                        ema_34_1w_aligned[i] > ema_34_1w_aligned[i-1])
            
            # Short: break below S3 with volume spike and weekly downtrend
            short_cond = (close[i] < S3_12h[i] and 
                         volume_spike[i] and
                         ema_34_1w_aligned[i] < ema_34_1w_aligned[i-1])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price drops below S3 (reversal signal)
            if close[i] < S3_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price rises above R3 (reversal signal)
            if close[i] > R3_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals