#!/usr/bin/env python3
"""
1d_Pivot_R1S1_Breakout_WeeklyEMA_Filter_v1
Hypothesis: Use daily Camarilla pivot R1/S1 breakouts with volume confirmation and weekly EMA trend filter. In bull markets, buy R1 breakouts above weekly EMA; in bear markets, short S1 breakdowns below weekly EMA. This structure works in both regimes by following the higher timeframe trend. Weekly EMA ensures we only take trades aligned with the major trend, reducing whipsaws. Target: 10-25 trades/year to minimize fee drag.
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
    
    # Calculate daily Camarilla pivot levels (using previous day's HLC)
    # R1 = close + 1.1*(high - low)/12
    # S1 = close - 1.1*(high - low)/12
    # We need previous day's data, so shift by 1
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = high[0]  # fill first value
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    pivot_range = prev_high - prev_low
    R1 = prev_close + 1.1 * pivot_range / 12
    S1 = prev_close - 1.1 * pivot_range / 12
    
    # Weekly EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Volume confirmation: >1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 21  # Need EMA warmup
    
    for i in range(start_idx, n):
        if (np.isnan(ema_1w_aligned[i]) or 
            np.isnan(volume_spike[i]) or
            np.isnan(R1[i]) or
            np.isnan(S1[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema_1w_val = ema_1w_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: price breaks above R1 with volume and above weekly EMA
            if price > R1[i] and vol_spike and price > ema_1w_val:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume and below weekly EMA
            elif price < S1[i] and vol_spike and price < ema_1w_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price breaks below S1 (reversal) or below weekly EMA
            if price < S1[i] or price < ema_1w_val:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price breaks above R1 (reversal) or above weekly EMA
            if price > R1[i] or price > ema_1w_val:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Pivot_R1S1_Breakout_WeeklyEMA_Filter_v1"
timeframe = "1d"
leverage = 1.0