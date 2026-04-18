#!/usr/bin/env python3
"""
1h_12h_Pivot_R1S1_Breakout_4hTrend
Hypothesis: 12h pivot levels (R1, S1) act as strong support/resistance.
Breakouts beyond these levels on 1h chart with volume confirmation and 4h EMA50 trend filter capture momentum.
Designed for 15-37 trades/year on 1h timeframe by using 12h for signal direction and 4h for trend filter.
Works in bull/bear markets by requiring volume spike and 4h EMA50 trend filter.
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
    
    # Get 12h data for pivot calculation (once before loop)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h pivot points using standard formula
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    pivot = (high_12h + low_12h + close_12h) / 3
    r1 = 2 * pivot - low_12h
    s1 = 2 * pivot - high_12h
    
    # Shift by 1 to use previous 12h bar's levels only
    r1_prev = np.roll(r1, 1)
    s1_prev = np.roll(s1, 1)
    r1_prev[0] = np.nan
    s1_prev[0] = np.nan
    
    # Align to 1h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_12h, r1_prev)
    s1_aligned = align_htf_to_ltf(prices, df_12h, s1_prev)
    
    # Get 4h data for EMA trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Volume spike: 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or
            np.isnan(ema_50_4h_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        ema_trend = ema_50_4h_aligned[i]
        
        if position == 0:
            # Long: break above R1 with volume spike and price above 4h EMA (uptrend)
            if price > r1_val and volume_spike[i] and price > ema_trend:
                signals[i] = 0.20
                position = 1
            # Short: break below S1 with volume spike and price below 4h EMA (downtrend)
            elif price < s1_val and volume_spike[i] and price < ema_trend:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long position
            signals[i] = 0.20
            # Exit: price returns to S1 or breaks below 4h EMA
            if price <= s1_val or price < ema_trend:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position
            signals[i] = -0.20
            # Exit: price returns to R1 or breaks above 4h EMA
            if price >= r1_val or price > ema_trend:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_12h_Pivot_R1S1_Breakout_4hTrend"
timeframe = "1h"
leverage = 1.0