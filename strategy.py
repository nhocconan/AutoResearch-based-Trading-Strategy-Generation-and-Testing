#!/usr/bin/env python3
"""
12h_Camarilla_R1S1_Breakout_1wEMA50_Trend_VolumeSpike
Hypothesis: Camarilla R1/S1 levels from 1d act as key intraday support/resistance; breakouts with volume spike and 1w EMA50 trend filter capture institutional moves. Works in both bull (breakouts above R1 in uptrend) and bear (breakdowns below S1 in downtrend) regimes. 12h timeframe minimizes fee drag while allowing sufficient trade frequency.
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
    
    # 1d data for Camarilla levels (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels from previous 1d bar
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # Use previous day's close, high, low
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    camarilla_r1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    camarilla_s1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    
    # Align to 12h timeframe (wait for completed 1d bar)
    r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # 1w EMA50 for trend filter (loaded ONCE)
    df_1w = get_htf_data(prices, '1w')
    ema_50 = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Volume spike: current volume > 2.0 * 20-period volume MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need 1d Camarilla (shift 1), 1w EMA50 (50), volume MA (20)
    start_idx = max(50, 20, 1) + 5  # Conservative warmup
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        if position == 0:
            # Long: price breaks above Camarilla R1 with volume spike and 1w uptrend
            long_breakout = (curr_close > r1_aligned[i]) and vol_spike[i] and (curr_close > ema_aligned[i])
            # Short: price breaks below Camarilla S1 with volume spike and 1w downtrend
            short_breakout = (curr_close < s1_aligned[i]) and vol_spike[i] and (curr_close < ema_aligned[i])
            
            if long_breakout:
                signals[i] = 0.25
                position = 1
            elif short_breakout:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price breaks below Camarilla S1 or trend turns down
            if (curr_close < s1_aligned[i]) or (curr_close < ema_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above Camarilla R1 or trend turns up
            if (curr_close > r1_aligned[i]) or (curr_close > ema_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_1wEMA50_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0