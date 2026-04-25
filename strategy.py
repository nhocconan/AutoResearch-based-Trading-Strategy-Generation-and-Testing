#!/usr/bin/env python3
"""
12h_Camarilla_R1S1_Breakout_1wTrend_VolumeSpike
Hypothesis: On 12h timeframe, Camarilla pivot levels (R1/S1) from the previous week capture institutional interest.
Break above R1 with volume spike and 1w uptrend (close > EMA34) signals long; break below S1 with volume spike and 
1w downtrend (close < EMA34) signals short. Uses discrete position sizing (0.25) to limit trades (~12-37/year) and 
minimize fee drag. Designed for BTC/ETH to work in both bull and bear markets by trading breakouts with trend and 
volume confirmation, avoiding overtrading through tight entry conditions.
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
    
    # 1w data for trend filter (loaded ONCE)
    df_1w = get_htf_data(prices, '1w')
    
    # 1w EMA34 for trend filter (loaded ONCE)
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Previous week's Camarilla levels (R1, S1)
    # Camarilla: R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    camarilla_r1 = close_1w + 1.1 * (high_1w - low_1w) / 12
    camarilla_s1 = close_1w - 1.1 * (high_1w - low_1w) / 12
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s1)
    
    # 12h volume spike: current volume > 2.0 * 20-period volume MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need volume MA (20) + aligned HTF arrays
    start_idx = max(20, 0)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        
        if position == 0:
            # Long: price breaks above Camarilla R1 with volume spike and 1w uptrend
            long_breakout = (curr_close > camarilla_r1_aligned[i]) and vol_spike[i] and (curr_close > ema_34_1w_aligned[i])
            # Short: price breaks below Camarilla S1 with volume spike and 1w downtrend
            short_breakout = (curr_close < camarilla_s1_aligned[i]) and vol_spike[i] and (curr_close < ema_34_1w_aligned[i])
            
            if long_breakout:
                signals[i] = 0.25
                position = 1
            elif short_breakout:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price breaks below Camarilla S1 OR trend turns down
            if (curr_close < camarilla_s1_aligned[i]) or (curr_close < ema_34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price breaks above Camarilla R1 OR trend turns up
            if (curr_close > camarilla_r1_aligned[i]) or (curr_close > ema_34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_1wTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0