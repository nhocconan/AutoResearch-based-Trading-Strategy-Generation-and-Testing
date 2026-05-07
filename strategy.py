# 12h_1W_Pivot_R1S1_Breakout_1DTrend_Volume
# Hypothesis: Breakouts above weekly S1 in daily uptrend or below weekly R1 in daily downtrend with volume confirmation capture institutional participation. Weekly pivot provides structure; daily trend filters direction; volume confirms strength. Designed for 12h timeframe to limit trades (12-37/year) and avoid fee drag. Works in bull (buy S1 breaks) and bear (sell R1 breaks).
# Timeframe: 12h, Target trades: 50-150 total over 4 years

#!/usr/bin/env python3
name = "12h_1W_Pivot_R1S1_Breakout_1DTrend_Volume"
timeframe = "12h"
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
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Weekly pivot points from daily data (use last completed week)
    # Approximate week as 7 days of daily data
    weekly_high = pd.Series(high).rolling(window=7, min_periods=7).max().values
    weekly_low = pd.Series(low).rolling(window=7, min_periods=7).min().values
    weekly_close = pd.Series(close).rolling(window=7, min_periods=7).mean().values
    
    # Pivot levels
    pp = (weekly_high + weekly_low + weekly_close) / 3
    r1 = 2 * pp - weekly_low
    s1 = 2 * pp - weekly_high
    
    # Align weekly pivot levels to 12h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp.values)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1.values)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1.values)
    
    # Daily trend filter: EMA(34) on daily close
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike detection: 2-period average (24h of 12h bars)
    vol_ma_2 = pd.Series(volume).rolling(window=2, min_periods=2).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 2, 7)  # Wait for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(pp_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(vol_ma_2[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above S1 with volume and daily uptrend
            vol_condition = volume[i] > vol_ma_2[i] * 2.0
            uptrend = ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]
            
            if close[i] > s1_aligned[i] and vol_condition and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: price below R1 with volume and daily downtrend
            elif close[i] < r1_aligned[i] and vol_condition and not uptrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price back below S1 or volume drops
            if close[i] < s1_aligned[i] or volume[i] < vol_ma_2[i] * 1.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price back above R1 or volume drops
            if close[i] > r1_aligned[i] or volume[i] < vol_ma_2[i] * 1.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Breakouts above weekly S1 in daily uptrend or below weekly R1 in daily downtrend with volume confirmation capture institutional participation. Weekly pivot provides structure; daily trend filters direction; volume confirms strength. Designed for 12h timeframe to limit trades (12-37/year) and avoid fee drag. Works in bull (buy S1 breaks) and bear (sell R1 breaks).