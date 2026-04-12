#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1h strategy using 4h Camarilla pivot breakouts with volume confirmation
    # Camarilla levels provide intraday support/resistance based on prior day's range
    # Breakouts above/below key levels with volume spike indicate institutional participation
    # Session filter (08-20 UTC) focuses on liquid London/NY overlap
    # Discrete position sizing (0.20) controls drawdown and minimizes fee churn
    # Target: 15-37 trades/year per symbol (60-150 total over 4 years)
    
    # Session filter: 8:00-20:00 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot calculation (based on prior day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate prior day's Camarilla levels
    # H4 = Close + 1.5*(High-Low)  (resistance)
    # L4 = Close - 1.5*(High-Low)  (support)
    # H3 = Close + 1.125*(High-Low)
    # L3 = Close - 1.125*(High-Low)
    # H2 = Close + 0.75*(High-Low)
    # L2 = Close - 0.75*(High-Low)
    # H1 = Close + 0.5*(High-Low)
    # L1 = Close - 0.5*(High-Low)
    # Pivot = (High + Low + Close)/3
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Prior day's range
    range_1d = high_1d - low_1d
    
    # Camarilla levels for prior day
    h4_1d = close_1d + 1.5 * range_1d
    l4_1d = close_1d - 1.5 * range_1d
    h3_1d = close_1d + 1.125 * range_1d
    l3_1d = close_1d - 1.125 * range_1d
    h2_1d = close_1d + 0.75 * range_1d
    l2_1d = close_1d - 0.75 * range_1d
    h1_1d = close_1d + 0.5 * range_1d
    l1_1d = close_1d - 0.5 * range_1d
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    
    # Align 1d Camarilla levels to 1h timeframe (prior day's levels available at 00:00 UTC)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4_1d)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4_1d)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3_1d)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3_1d)
    h2_aligned = align_htf_to_ltf(prices, df_1d, h2_1d)
    l2_aligned = align_htf_to_ltf(prices, df_1d, l2_1d)
    h1_aligned = align_htf_to_ltf(prices, df_1d, h1_1d)
    l1_aligned = align_htf_to_ltf(prices, df_1d, l1_1d)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    close_s = pd.Series(close)
    volume_s = pd.Series(volume)
    vol_ma_20 = volume_s.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 1.5 * vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if data not ready
        if (np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or 
            np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Breakout logic: price breaks key Camarilla levels with volume spike
        long_breakout = (close[i] > h3_aligned[i] or close[i] > h4_aligned[i]) and volume_spike[i]
        short_breakout = (close[i] < l3_aligned[i] or close[i] < l4_aligned[i]) and volume_spike[i]
        
        # Exit logic: price returns to pivot or opposite test of levels
        long_exit = close[i] < pivot_aligned[i]
        short_exit = close[i] > pivot_aligned[i]
        
        if long_breakout and position != 1:
            position = 1
            signals[i] = 0.20
        elif short_breakout and position != -1:
            position = -1
            signals[i] = -0.20
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
    
    return signals

name = "1h_1d_camarilla_breakout_vol_v1"
timeframe = "1h"
leverage = 1.0