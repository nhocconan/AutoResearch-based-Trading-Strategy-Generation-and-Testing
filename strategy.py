#!/usr/bin/env python3
"""
4h_Camarilla_Pivot_R1_S1_Breakout_Volume_Trend
Hypothesis: Camarilla pivot levels (R1, S1) from daily chart act as strong support/resistance.
Breakouts above R1 or below S1 with volume confirmation and 4h EMA trend filter capture
institutional breakout moves. Works in both bull (breakouts continue) and bear (breakdowns continue)
markets. Target: 20-35 trades/year (80-140 total over 4 years) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivots (calculated from previous day)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each day: based on previous day's OHLC
    # R1 = close + 1.1*(high - low)/12
    # S1 = close - 1.1*(high - low)/12
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Avoid look-ahead: use previous day's data only
    rang = prev_high - prev_low
    R1 = prev_close + 1.1 * rang / 12
    S1 = prev_close - 1.1 * rang / 12
    
    # Align to 4h timeframe (values update only at daily open)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # 4h EMA34 trend filter
    ema_34 = pd.Series(close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Volume filter: >1.6x 20-period average (slightly higher to reduce trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.6 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 35  # Warmup for EMA34 and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or
            np.isnan(ema_34[i]) or np.isnan(volume_filter[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        r1 = R1_aligned[i]
        s1 = S1_aligned[i]
        vol_ok = volume_filter[i]
        ema34 = ema_34[i]
        
        if position == 0:
            # Long: price breaks above R1 with volume in uptrend
            if price > r1 and vol_ok and price > ema34:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume in downtrend
            elif price < s1 and vol_ok and price < ema34:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price returns to S1 or trend reverses
            if price < s1 or price < ema34:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price returns to R1 or trend reverses
            if price > r1 or price > ema34:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_Pivot_R1_S1_Breakout_Volume_Trend"
timeframe = "4h"
leverage = 1.0