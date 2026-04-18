#!/usr/bin/env python3
"""
1d_Weekly_Pivot_R1S1_Breakout_With_Volume_and_WeeklyTrend
Hypothesis: Buy when price breaks above weekly R1 with volume spike and above weekly EMA34; short when breaks below S1 with volume spike and below weekly EMA34. Weekly pivots capture longer-term structure, reducing whipsaw. Volume confirms institutional participation, and weekly EMA34 ensures alignment with major trend. Designed for very low trade frequency (<20/year) to minimize fee drift while capturing high-probability breakouts in both bull and bear regimes.
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
    
    # Weekly OHLC for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    pweek_high = df_1w['high'].values
    pweek_low = df_1w['low'].values
    pweek_close = df_1w['close'].values
    
    # Weekly Camarilla levels
    rang = pweek_high - pweek_low
    r1 = pweek_close + rang * 1.1 / 12
    s1 = pweek_close - rang * 1.1 / 12
    
    # Align to daily timeframe (wait for weekly close)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    
    # Volume spike: >2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Weekly EMA34 trend filter
    weekly_close = df_1w['close'].values
    weekly_ema = pd.Series(weekly_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    weekly_ema_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 40  # Need volume MA and indicators
    
    for i in range(start_idx, n):
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or
            np.isnan(volume_spike[i]) or
            np.isnan(weekly_ema_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        vol_spike = volume_spike[i]
        weekly_ema_val = weekly_ema_aligned[i]
        
        if position == 0:
            # Long: price > R1 with volume spike and above weekly EMA34
            if price > r1_val and vol_spike and price > weekly_ema_val:
                signals[i] = 0.25
                position = 1
            # Short: price < S1 with volume spike and below weekly EMA34
            elif price < s1_val and vol_spike and price < weekly_ema_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price < S1 or below weekly EMA34
            if price < s1_val or price < weekly_ema_val:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price > R1 or above weekly EMA34
            if price > r1_val or price > weekly_ema_val:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Weekly_Pivot_R1S1_Breakout_With_Volume_and_WeeklyTrend"
timeframe = "1d"
leverage = 1.0