#!/usr/bin/env python3
"""
1d_WeeklyPivot_R1S1_Breakout_Volume_Confirmation
Hypothesis: Weekly pivot levels (R1/S1) on 1d timeframe provide strong support/resistance.
Breakout with volume confirmation and weekly trend filter (EMA34) captures institutional moves.
Works in bull/bear by following breakouts with confirmation. Targets 15-25 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Weekly pivot levels from previous week (1w timeframe)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Weekly Pivot calculation: R1/S1 from previous week
    # R1 = close + 1.1 * (high - low) / 12
    # S1 = close - 1.1 * (high - low) / 12
    weekly_range = high_1w - low_1w
    r1_1w = close_1w + (1.1 * weekly_range) / 12
    s1_1w = close_1w - (1.1 * weekly_range) / 12
    
    # Align to 1d timeframe (use previous week's levels)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    
    # Weekly EMA trend filter (34-period)
    ema_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Volume confirmation: >2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(ema_1w_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        r1 = r1_aligned[i]
        s1 = s1_aligned[i]
        ema_1w_val = ema_1w_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: price breaks above R1 with volume and above weekly EMA
            if price > r1 and vol_spike and price > ema_1w_val:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume and below weekly EMA
            elif price < s1 and vol_spike and price < ema_1w_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price below S1 or below weekly EMA
            if price < s1 or price < ema_1w_val:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price above R1 or above weekly EMA
            if price > r1 or price > ema_1w_val:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_WeeklyPivot_R1S1_Breakout_Volume_Confirmation"
timeframe = "1d"
leverage = 1.0