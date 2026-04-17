#!/usr/bin/env python3
"""
6h_Pivot_R1_S1_Breakout_Volume_WeeklyTrend_Filter
Strategy: 6h breakout above R1 or below S1 with volume confirmation, filtered by weekly trend.
Uses 1d Pivot Points (R1/S1) and 1w EMA(20) for trend filter.
Long: break above R1 + volume > 1.5x avg + price > weekly EMA20
Short: break below S1 + volume > 1.5x avg + price < weekly EMA20
Exit: return to pivot point (PP) or trend reversal
Position size: 0.25
Designed to capture institutional breakouts with weekly trend alignment.
Timeframe: 6h
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 20-period volume MA for confirmation
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get 1d data for pivot points (using previous day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate pivot points from previous day's OHLC
    # Pivot Point (PP) = (High + Low + Close) / 3
    # R1 = 2*PP - Low
    # S1 = 2*PP - High
    pp = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    r1 = 2 * pp - df_1d['low']
    s1 = 2 * pp - df_1d['high']
    
    # Align pivot levels to 6h timeframe (they represent previous day's levels)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp.values)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1.values)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1.values)
    
    # Get 1w data for trend filter (EMA20)
    df_1w = get_htf_data(prices, '1w')
    # Calculate weekly EMA20
    ema_20_1w = pd.Series(df_1w['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    # Align weekly EMA to 6h timeframe
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)  # Need volume MA20 and weekly EMA warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(volume_ma20[i]) or 
            np.isnan(ema_20_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period average
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        # Entry conditions
        if position == 0:
            # Long: break above R1 + volume + above weekly EMA20
            if (close[i] > r1_aligned[i] and 
                volume_filter and 
                close[i] > ema_20_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below S1 + volume + below weekly EMA20
            elif (close[i] < s1_aligned[i] and 
                  volume_filter and 
                  close[i] < ema_20_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: return to pivot point or trend reversal (below weekly EMA)
            if close[i] <= pp_aligned[i] or close[i] < ema_20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: return to pivot point or trend reversal (above weekly EMA)
            if close[i] >= pp_aligned[i] or close[i] > ema_20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Pivot_R1_S1_Breakout_Volume_WeeklyTrend_Filter"
timeframe = "6h"
leverage = 1.0