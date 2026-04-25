#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_12hEMA34_Trend_VolumeSpike
Hypothesis: Camarilla pivot levels (R1/S1) from 1d act as key intraday support/resistance. A break of these levels with volume spike and alignment with 12h EMA34 trend captures strong momentum moves. Works in bull markets (breakouts continuation) and bear markets (breakdown continuation). Uses discrete position sizing (0.25) to limit fee drag and drawdown.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given high, low, close."""
    pivot = (high + low + close) / 3.0
    range_ = high - low
    r1 = close + range_ * 1.1 / 12.0
    s1 = close - range_ * 1.1 / 12.0
    r2 = close + range_ * 1.1 / 6.0
    s2 = close - range_ * 1.1 / 6.0
    r3 = close + range_ * 1.1 / 4.0
    s3 = close - range_ * 1.1 / 4.0
    r4 = close + range_ * 1.1 / 2.0
    s4 = close - range_ * 1.1 / 2.0
    return pivot, r1, r2, r3, r4, s1, s2, s3, s4

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for Camarilla pivot levels (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    
    # 12h data for EMA34 trend filter (loaded ONCE)
    df_12h = get_htf_data(prices, '12h')
    
    # 1d Camarilla levels
    _, r1_1d, _, _, _, s1_1d, _, _, _ = calculate_camarilla(
        df_1d['high'].values, df_1d['low'].values, df_1d['close'].values
    )
    
    # 12h EMA34 for trend filter
    ema_34_12h = pd.Series(df_12h['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align HTF indicators to LTF (4h)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # 4h volume spike: current volume > 2.0 * 20-period volume MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need volume MA (20) + aligned HTF arrays
    start_idx = max(20, 0)  # align_htf_to_ltf handles warmup internally
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or 
            np.isnan(ema_34_12h_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        if position == 0:
            # Long: price breaks above R1 with volume spike and 12h uptrend
            long_breakout = (curr_close > r1_1d_aligned[i]) and vol_spike[i] and (curr_close > ema_34_12h_aligned[i])
            # Short: price breaks below S1 with volume spike and 12h downtrend
            short_breakout = (curr_close < s1_1d_aligned[i]) and vol_spike[i] and (curr_close < ema_34_12h_aligned[i])
            
            if long_breakout:
                signals[i] = 0.25
                position = 1
            elif short_breakout:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price breaks below S1 or trend turns down
            if (curr_close < s1_1d_aligned[i]) or (curr_close < ema_34_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above R1 or trend turns up
            if (curr_close > r1_1d_aligned[i]) or (curr_close > ema_34_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_12hEMA34_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0