#!/usr/bin/env python3
"""
1d_1W_Camarilla_R1S1_Breakout_Volume_Sparse_v1
Hypothesis: Daily chart breakout above weekly Camarilla R1/S1 levels with volume confirmation.
Uses weekly trend filter (price above/below weekly EMA34) to avoid counter-trend trades.
Targets low frequency (5-15 trades/year) by requiring confluence of daily breakout, volume spike,
and weekly trend alignment. Works in bull/bear via weekly trend filter.
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
    
    # Get weekly data for trend filter and context
    df_1w = get_htf_data(prices, '1w')
    
    # Get daily data for primary signal generation
    df_1d = get_htf_data(prices, '1d')
    
    # Weekly calculations for trend filter
    close_1w = df_1w['close'].values
    
    # Weekly EMA34 for trend direction
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Daily calculations for signal
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Previous day's OHLC for Camarilla calculation
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close[0] = close_1d[0]
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    
    # Daily Camarilla levels: R1 = close + (high-low)*1.1/12, S1 = close - (high-low)*1.1/12
    range_1d = prev_high - prev_low
    r1_1d = prev_close + range_1d * 1.1 / 12
    s1_1d = prev_close - range_1d * 1.1 / 12
    
    # Align daily Camarilla levels to 1d timeframe (no shift needed as we use previous day's levels)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Precompute volume confirmation: current volume > 2.0x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > 2.0 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # need enough for EMA and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or 
            np.isnan(ema34_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above daily R1 with volume confirmation and weekly uptrend
            if (close[i] > r1_1d_aligned[i] and vol_confirm[i] and 
                close[i] > ema34_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below daily S1 with volume confirmation and weekly downtrend
            elif (close[i] < s1_1d_aligned[i] and vol_confirm[i] and 
                  close[i] < ema34_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns below daily R1 or weekly trend turns down
            if close[i] < r1_1d_aligned[i] or close[i] < ema34_1w_aligned[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns above daily S1 or weekly trend turns up
            if close[i] > s1_1d_aligned[i] or close[i] > ema34_1w_aligned[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_1W_Camarilla_R1S1_Breakout_Volume_Sparse_v1"
timeframe = "1d"
leverage = 1.0