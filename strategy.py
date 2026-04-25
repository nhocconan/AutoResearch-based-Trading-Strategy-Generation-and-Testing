#!/usr/bin/env python3
"""
4h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike
Hypothesis: Camarilla pivot levels (R3/S3) from 1d act as strong support/resistance. 
Breakout above R3 or below S3 with volume spike and 1d EMA34 trend filter captures 
institutional participation in both bull and bear regimes. Uses discrete position 
sizing (0.25) to minimize fee churn and allow for partial exits.
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
    
    # 1d data for Camarilla pivots, EMA trend (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    
    # Camarilla pivot levels for 1d
    # R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low), etc.
    # We use R3 and S3 as breakout levels
    hl_range = df_1d['high'].values - df_1d['low'].values
    close_1d = df_1d['close'].values
    r3 = close_1d + 1.1 * hl_range
    s3 = close_1d - 1.1 * hl_range
    # Align to 4h timeframe (wait for 1d bar to close)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # 1d EMA34 for trend filter
    ema_34 = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # 4h volume spike: current volume > 2.0 * 20-period volume MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need 1d EMA (34) + volume MA (20)
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        if position == 0:
            # Long: price breaks above R3 (resistance) with volume spike and 1d uptrend
            long_breakout = (curr_close > r3_aligned[i]) and vol_spike[i] and (curr_close > ema_aligned[i])
            # Short: price breaks below S3 (support) with volume spike and 1d downtrend
            short_breakout = (curr_close < s3_aligned[i]) and vol_spike[i] and (curr_close < ema_aligned[i])
            
            if long_breakout:
                signals[i] = 0.25
                position = 1
            elif short_breakout:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price breaks below R3 or trend turns down
            if (curr_close < r3_aligned[i]) or (curr_close < ema_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above S3 or trend turns up
            if (curr_close > s3_aligned[i]) or (curr_close > ema_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike"
timeframe = "4h"
leverage = 1.0