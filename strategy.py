#!/usr/bin/env python3
"""
12h_1d_Camarilla_Pivot_Breakout_Volume_Trend_v1
Hypothesis: On 12h timeframe, take long when price breaks above Camarilla H3 (bullish breakout) 
with volume confirmation and weekly trend alignment, and short when price breaks below L3 (bearish breakout)
with volume confirmation and weekly trend alignment. Uses 1d Camarilla levels calculated from prior day,
volume spike filter, and 1w EMA50 trend filter to avoid counter-trend trades. Designed for 12-37 trades/year
by requiring confluence: Camarilla breakout, volume spike, and trend alignment. Works in bull markets via 
long breakouts and in bear markets via short breakdowns.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_Camarilla_Pivot_Breakout_Volume_Trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivot levels from prior day's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Prior day's values (shift by 1 to avoid look-ahead)
    high_prev = np.roll(high_1d, 1)
    low_prev = np.roll(low_1d, 1)
    close_prev = np.roll(close_1d, 1)
    high_prev[0] = high_1d[0]  # First bar uses same day
    low_prev[0] = low_1d[0]
    close_prev[0] = close_1d[0]
    
    # Camarilla calculations
    range_prev = high_prev - low_prev
    H3 = close_prev + range_prev * 1.1 / 4
    L3 = close_prev - range_prev * 1.1 / 4
    H4 = close_prev + range_prev * 1.1 / 2
    L4 = close_prev - range_prev * 1.1 / 2
    
    # Align Camarilla levels to 12h timeframe
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    
    # Load 1w data ONCE for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate 1w EMA50
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume average (24 period = 12 days) for spike detection
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(close[i]) or np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Breakout conditions
        breakout_long = close[i] > H3_aligned[i]  # Break above H3
        breakout_short = close[i] < L3_aligned[i]  # Break below L3
        
        # Volume spike: current volume > 2.0x average
        volume_spike = volume[i] > vol_ma[i] * 2.0
        
        # Trend filter from 1w EMA50
        above_ema = close[i] > ema_50_1w_aligned[i]
        below_ema = close[i] < ema_50_1w_aligned[i]
        
        # Entry conditions with confluence
        long_entry = breakout_long and volume_spike and above_ema
        short_entry = breakout_short and volume_spike and below_ema
        
        # Exit conditions: reversal to opposite Camarilla level or trend change
        long_exit = close[i] < L3_aligned[i] or close[i] < ema_50_1w_aligned[i]
        short_exit = close[i] > H3_aligned[i] or close[i] > ema_50_1w_aligned[i]
        
        # Priority: entry > exit > hold
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals