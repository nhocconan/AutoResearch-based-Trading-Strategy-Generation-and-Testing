#!/usr/bin/env python3
"""
1d_Weekly_Pivot_R1S1_Breakout_With_Volume_Confirmation
Hypothesis: Price breaks above/below weekly pivot S1/R1 with volume spike and daily trend filter.
Weekly pivots provide strong support/resistance levels; breakouts with volume indicate institutional interest.
Daily trend filter (EMA50) ensures alignment with higher timeframe momentum. Designed for 1d timeframe
to target 15-25 trades/year, minimizing fee drag while capturing significant moves in bull/bear markets.
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
    
    # Weekly pivot from previous week
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Pivot levels: P = (H+L+C)/3, R1 = C + (H-L)*1.1/2, S1 = C - (H-L)*1.1/2
    pivot = (high_1w + low_1w + close_1w) / 3.0
    r1 = close_1w + (high_1w - low_1w) * 1.1 / 2.0
    s1 = close_1w - (high_1w - low_1w) * 1.1 / 2.0
    
    # Align to 1d: previous week's levels available after weekly bar closes
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    
    # Volume spike: >2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Trend filter: daily EMA50
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(50, 20)  # Warmup for EMA and indicators
    
    for i in range(start_idx, n):
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or
            np.isnan(ema_50[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        ema50 = ema_50[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: price breaks above R1 with volume spike and uptrend
            if price > r1_val and vol_spike and price > ema50:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume spike and downtrend
            elif price < s1_val and vol_spike and price < ema50:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price closes below weekly pivot OR trend turns down
            if price < pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            elif price < ema50:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price closes above weekly pivot OR trend turns up
            if price > pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            elif price > ema50:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Weekly_Pivot_R1S1_Breakout_With_Volume_Confirmation"
timeframe = "1d"
leverage = 1.0