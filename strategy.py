#!/usr/bin/env python3
"""
6h_WeeklyPivot_Pullback_WithVolume
Hypothesis: Buy pullbacks to weekly pivot support in uptrends, sell rallies to weekly pivot resistance in downtrends.
Uses weekly pivot as structural support/resistance, EMA50 for trend filter, volume surge for confirmation.
Designed for 6h timeframe to target 50-150 total trades over 4 years (12-37/year).
Works in bull/bear via trend filter and mean-reversion at key weekly levels.
"""

name = "6h_WeeklyPivot_Pullback_WithVolume"
timeframe = "6h"
leverage = 1.0

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
    
    # Weekly data for pivot calculation (trend filter and reference)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Weekly OHLC for pivot points
    wk_high = df_1w['high'].values
    wk_low = df_1w['low'].values
    wk_close = df_1w['close'].values
    
    # Weekly pivot point calculation (standard floor trader's pivot)
    wk_p = (wk_high + wk_low + wk_close) / 3.0
    wk_r1 = 2 * wk_p - wk_low
    wk_s1 = 2 * wk_p - wk_high
    wk_r2 = wk_p + (wk_high - wk_low)
    wk_s2 = wk_p - (wk_high - wk_low)
    
    # Align weekly pivot levels to 6h timeframe
    wk_p_aligned = align_htf_to_ltf(prices, df_1w, wk_p)
    wk_r1_aligned = align_htf_to_ltf(prices, df_1w, wk_r1)
    wk_s1_aligned = align_htf_to_ltf(prices, df_1w, wk_s1)
    wk_r2_aligned = align_htf_to_ltf(prices, df_1w, wk_r2)
    wk_s2_aligned = align_htf_to_ltf(prices, df_1w, wk_s2)
    
    # EMA50 for trend filter (calculated on 6h data)
    close_series = pd.Series(close)
    ema50 = close_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume confirmation: volume > 2.0 * 20-period average (strict for fewer trades)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Ensure enough data for EMA50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema50[i]) or np.isnan(wk_p_aligned[i]) or np.isnan(wk_r1_aligned[i]) or 
            np.isnan(wk_s1_aligned[i]) or np.isnan(wk_r2_aligned[i]) or np.isnan(wk_s2_aligned[i]) or 
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price > EMA50 for uptrend, price < EMA50 for downtrend
        uptrend = close[i] > ema50[i]
        downtrend = close[i] < ema50[i]
        
        if position == 0:
            # Long: pullback to weekly S1 in uptrend with volume confirmation
            if (uptrend and 
                low[i] <= wk_s1_aligned[i] and  # Price touches or goes below S1
                close[i] > wk_s1_aligned[i] and   # But closes back above S1 (pullback confirmation)
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: rally to weekly R1 in downtrend with volume confirmation
            elif (downtrend and 
                  high[i] >= wk_r1_aligned[i] and  # Price touches or goes above R1
                  close[i] < wk_r1_aligned[i] and   # But closes back below R1 (rejection confirmation)
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price reaches weekly R1 (take profit) or trend breaks down
            if (close[i] >= wk_r1_aligned[i]) or (not uptrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price reaches weekly S1 (take profit) or trend breaks up
            if (close[i] <= wk_s1_aligned[i]) or (not downtrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals