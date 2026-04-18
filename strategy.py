#!/usr/bin/env python3
"""
12h_Pivot_R1S1_R2S2_Breakout_Volume_Trend
Hypothesis: Price breaks above/below Camarilla pivot levels (R1/S1) from the prior day with volume confirmation and 1-day EMA trend filter.
Camarilla levels derived from prior day's high, low, close. Breakouts require volume > 1.5x 20-period average and price > EMA34(1d) for longs,
price < EMA34(1d) for shorts. Designed to capture volatility expansion moves with tight entry conditions to minimize fee drift.
Target: 12-37 trades per year (50-150 total over 4 years).
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
    
    # Load 1-day data once before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels from prior day's OHLC
    # R1 = close + (high - low) * 1.1 / 12
    # S1 = close - (high - low) * 1.1 / 12
    # R2 = close + (high - low) * 1.1 / 6
    # S2 = close - (high - low) * 1.1 / 6
    # We'll use R1/S1 for breakout entries
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot levels for each 1d bar
    range_1d = high_1d - low_1d
    R1 = close_1d + range_1d * 1.1 / 12
    S1 = close_1d - range_1d * 1.1 / 12
    R2 = close_1d + range_1d * 1.1 / 6
    S2 = close_1d - range_1d * 1.1 / 6
    
    # Align to 12h timeframe (wait for prior day's close)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    R2_aligned = align_htf_to_ltf(prices, df_1d, R2)
    S2_aligned = align_htf_to_ltf(prices, df_1d, S2)
    
    # EMA34 on 1-day close for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume filter: >1.5x 20-period average (12h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 20  # Warmup for volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or
            np.isnan(ema_34_aligned[i]) or np.isnan(volume_filter[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        r1 = R1_aligned[i]
        s1 = S1_aligned[i]
        ema34 = ema_34_aligned[i]
        vol_ok = volume_filter[i]
        
        if position == 0:
            # Long: price breaks above R1 with volume in uptrend (price > EMA34)
            if price > r1 and vol_ok and price > ema34:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume in downtrend (price < EMA34)
            elif price < s1 and vol_ok and price < ema34:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price returns to S1 or trend reverses (price < EMA34)
            if price < s1 or price < ema34:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price returns to R1 or trend reverses (price > EMA34)
            if price > r1 or price > ema34:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Pivot_R1S1_R2S2_Breakout_Volume_Trend"
timeframe = "12h"
leverage = 1.0