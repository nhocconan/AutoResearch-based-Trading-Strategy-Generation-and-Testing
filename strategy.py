#!/usr/bin/env python3
"""
1h_Camarilla_R1S1_Breakout_With_Volume_Trend
Hypothesis: Price breaks above/below S1/R1 levels with volume confirmation and 4h trend filter.
Uses 1d Camarilla pivot levels (R1/S1), volume > 1.5x 20-period average, and 4h EMA34 trend filter.
Designed to work in both bull and bear markets by requiring trend alignment.
Target: 20-30 trades/year to minimize fee drag while capturing institutional breakout moves.
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
    
    # Daily Camarilla pivot levels (calculated from previous day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels: R1, S1 based on previous day
    # R1 = close + 1.1*(high-low)/12
    # S1 = close - 1.1*(high-low)/12
    camarilla_range = (high_1d - low_1d) * 1.1 / 12
    r1 = close_1d + camarilla_range
    s1 = close_1d - camarilla_range
    
    # Align to 1h timeframe (use previous day's levels for current day)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # 4h EMA34 for trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Volume spike: >1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Session filter: 8-20 UTC (active trading hours)
    hour_index = prices.index.hour
    session_filter = (hour_index >= 8) & (hour_index <= 20)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(20, 1)  # Warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(ema_34_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if not session_filter[i]:
            signals[i] = 0.0
            continue
        
        price = close[i]
        r1_level = r1_aligned[i]
        s1_level = s1_aligned[i]
        ema34 = ema_34_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: price breaks above S1 with volume spike in uptrend
            if (price > s1_level and          # breaks above S1
                vol_spike and                 # volume confirmation
                price > ema34):               # uptrend filter
                signals[i] = 0.20
                position = 1
            # Short: price breaks below R1 with volume spike in downtrend
            elif (price < r1_level and        # breaks below R1
                  vol_spike and               # volume confirmation
                  price < ema34):             # downtrend filter
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            signals[i] = 0.20
            # Exit: price crosses back below S1 or trend reverses
            if price < s1_level or price < ema34:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.20
            # Exit: price crosses back above R1 or trend reverses
            if price > r1_level or price > ema34:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_Camarilla_R1S1_Breakout_With_Volume_Trend"
timeframe = "1h"
leverage = 1.0