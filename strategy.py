#!/usr/bin/env python3
"""
1d_Camarilla_Pivot_MeanReversion_WeeklyTrend
Hypothesis: Mean-reversion at daily Camarilla pivot levels (H4/L4) with weekly trend filter and volume confirmation. 
Long when price touches H4 in weekly downtrend with volume spike, short when price touches L4 in weekly uptrend with volume spike.
Exit at daily pivot point (PP). Designed for 1d timeframe to capture reversals in ranging markets with trend alignment.
Works in bull/bear via weekly trend filter and mean-reversion logic.
"""

name = "1d_Camarilla_Pivot_MeanReversion_WeeklyTrend"
timeframe = "1d"
leverage = 1.0

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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Get daily data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Previous day's OHLC for CAMARILLA calculation (H4/L4 levels)
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Calculate CAMARILLA pivot levels (focus on H4 and L4 for mean reversion)
    pp = (prev_high + prev_low + prev_close) / 3.0
    range_val = prev_high - prev_low
    h4 = pp + range_val * 1.1 / 2  # H4 level
    l4 = pp - range_val * 1.1 / 2  # L4 level
    
    # Align CAMARILLA levels to daily timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    
    # Weekly trend filter using close vs EMA20
    weekly_close = df_1w['close'].values
    weekly_ema20 = pd.Series(weekly_close).ewm(span=20, adjust=False, min_periods=20).mean().values
    weekly_ema20_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema20)
    weekly_close_aligned = align_htf_to_ltf(prices, df_1w, weekly_close)
    
    # Volume confirmation: 10-day average
    vol_ma10 = pd.Series(volume).rolling(window=10, min_periods=10).mean().values
    vol_ratio = np.divide(volume, vol_ma10, out=np.zeros_like(volume), where=vol_ma10!=0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Warmup for weekly EMA20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or np.isnan(pp_aligned[i]) or
            np.isnan(weekly_ema20_aligned[i]) or np.isnan(weekly_close_aligned[i]) or
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Weekly trend determination
        weekly_trend_up = weekly_close_aligned[i] > weekly_ema20_aligned[i]
        weekly_trend_down = weekly_close_aligned[i] < weekly_ema20_aligned[i]
        
        if position == 0:
            # Long: price touches H4 in weekly downtrend with volume spike (mean reversion down)
            if (low[i] <= h4_aligned[i] and  # touched or penetrated H4
                weekly_trend_down and 
                vol_ratio[i] > 1.8):
                signals[i] = 0.25
                position = 1
            # Short: price touches L4 in weekly uptrend with volume spike (mean reversion up)
            elif (high[i] >= l4_aligned[i] and  # touched or penetrated L4
                  weekly_trend_up and 
                  vol_ratio[i] > 1.8):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to pivot point (mean reversion complete)
            if close[i] >= pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to pivot point (mean reversion complete)
            if close[i] <= pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals