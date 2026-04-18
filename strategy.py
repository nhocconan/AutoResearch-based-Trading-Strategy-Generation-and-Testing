#!/usr/bin/env python3
"""
1D_1W_Camarilla_R1S1_Breakout_Volume_Momentum_v1
Hypothesis: Use weekly trend direction (price above/below weekly EMA200) as primary bias, then enter long when price breaks above daily R1 with volume > 1.8x average during 08-20 UTC, short when breaks below daily S1 with volume > 1.8x average. Uses weekly EMA200 to avoid counter-trend trades in strong trends, reducing whipsaw. Target: 15-25 trades/year per symbol (60-100 total over 4 years) to minimize fee drag. Works in bull via weekly uptrend bias, in bear via weekly downtrend bias.
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
    
    # Get daily data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Previous day's OHLC for Camarilla calculation
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close[0] = close_1d[0]  # first day uses same day
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    
    # Camarilla levels: R1 = close + (high-low)*1.1/12, S1 = close - (high-low)*1.1/12
    range_1d = prev_high - prev_low
    r1 = prev_close + range_1d * 1.1 / 12
    s1 = prev_close - range_1d * 1.1 / 12
    
    # Get weekly data for trend filter (EMA200)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align all data to daily timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # need enough for weekly EMA200
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema200_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.8x 20-period average
        vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_confirm = volume[i] > 1.8 * vol_ma[i] if not np.isnan(vol_ma[i]) else False
        
        # Only trade during active session
        in_session = session_mask[i]
        
        if position == 0:
            # Long: price breaks above R1, price above weekly EMA200 (uptrend), volume confirmation during session
            if close[i] > r1_aligned[i] and close[i] > ema200_1w_aligned[i] and vol_confirm and in_session:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1, price below weekly EMA200 (downtrend), volume confirmation during session
            elif close[i] < s1_aligned[i] and close[i] < ema200_1w_aligned[i] and vol_confirm and in_session:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns below R1 or below weekly EMA200 (trend change) or outside session
            if close[i] < r1_aligned[i] or close[i] < ema200_1w_aligned[i] or not in_session:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns above S1 or above weekly EMA200 (trend change) or outside session
            if close[i] > s1_aligned[i] or close[i] > ema200_1w_aligned[i] or not in_session:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "1D_1W_Camarilla_R1S1_Breakout_Volume_Momentum_v1"
timeframe = "1d"
leverage = 1.0