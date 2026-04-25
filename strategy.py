#!/usr/bin/env python3
"""
12h_Camarilla_R1S1_Breakout_1dTrend_VolumeSpike
Hypothesis: Camarilla R1/S1 levels from 1d act as key support/resistance. A break above R1 with volume spike and 1d uptrend signals long; break below S1 with volume spike and 1d downtrend signals short. Works in both bull and bear markets by capturing institutional breakout levels with volume confirmation and trend filter. Target: 12-37 trades/year per symbol.
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
    
    # 1d data for Camarilla pivot levels and EMA34 trend (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels from previous 1d bar
    # Using high, low, close of completed 1d bar
    h = df_1d['high'].values
    l = df_1d['low'].values
    c = df_1d['close'].values
    
    # Camarilla levels: R4, R3, R2, R1, PP, S1, S2, S3, S4
    # We only need R1 and S1 for breakout
    # R1 = C + (H-L)*1.1/12
    # S1 = C - (H-L)*1.1/12
    camarilla_r1 = c + (h - l) * 1.1 / 12
    camarilla_s1 = c - (h - l) * 1.1 / 12
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align HTF indicators to LTF (12h) - Camarilla levels need no extra delay as they're based on completed 1d bar
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 12h volume spike: current volume > 2.0 * 20-period volume MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for stoploss
    
    # Start index: need volume MA (20) + aligned HTF arrays
    start_idx = max(20, 0)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        if position == 0:
            # Long: price breaks above Camarilla R1 with volume spike and 1d uptrend
            long_breakout = (curr_close > camarilla_r1_aligned[i]) and vol_spike[i] and (curr_close > ema_34_1d_aligned[i])
            # Short: price breaks below Camarilla S1 with volume spike and 1d downtrend
            short_breakout = (curr_close < camarilla_s1_aligned[i]) and vol_spike[i] and (curr_close < ema_34_1d_aligned[i])
            
            if long_breakout:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_breakout:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price breaks below Camarilla S1 OR trend turns down
            if (curr_close < camarilla_s1_aligned[i]) or (curr_close < ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price breaks above Camarilla R1 OR trend turns up
            if (curr_close > camarilla_r1_aligned[i]) or (curr_close > ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_1dTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0