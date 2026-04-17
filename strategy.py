#!/usr/bin/env python3
"""
1d_WeeklyPivot_Squeeze_Breakout
Strategy: Weekly Pivot Point breakout with Bollinger Squeeze filter.
- Long when price breaks above weekly R1 with Bollinger Bandwidth < 50th percentile
- Short when price breaks below weekly S1 with Bollinger Bandwidth < 50th percentile
- Exit when price returns to weekly pivot (PP) or volatility expands
Position size: 0.25
Designed to capture breakouts from low volatility regimes using weekly structure.
Timeframe: 1d
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_hlf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate weekly pivot points from weekly data
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) == 0:
        return np.zeros(n)
    
    # Weekly high, low, close
    wh = df_weekly['high'].values
    wl = df_weekly['low'].values
    wc = df_weekly['close'].values
    
    # Weekly pivot point and support/resistance levels
    pp = (wh + wl + wc) / 3.0
    r1 = 2 * pp - wl
    s1 = 2 * pp - wh
    r2 = pp + (wh - wl)
    s2 = pp - (wh - wl)
    
    # Align weekly levels to daily
    pp_aligned = align_htf_to_ltf(prices, df_weekly, pp)
    r1_aligned = align_htf_to_ltf(prices, df_weekly, r1)
    s1_aligned = align_htf_to_ltf(prices, df_weekly, s1)
    
    # Bollinger Bands (20, 2) on daily
    close_series = pd.Series(close)
    basis = close_series.rolling(window=20, min_periods=20).mean()
    dev = close_series.rolling(window=20, min_periods=20).std()
    upper = basis + 2.0 * dev
    lower = basis - 2.0 * dev
    bandwidth = (upper - lower) / basis
    
    # Percentile rank of bandwidth (252 lookback ~ 1 year)
    bandwidth_series = pd.Series(bandwidth)
    bandwidth_percentile = bandwidth_series.rolling(window=252, min_periods=50).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5, raw=False
    ).values
    
    # Squeeze condition: bandwidth below 50th percentile
    squeeze = bandwidth_percentile < 0.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 252)  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(squeeze[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions
        if position == 0:
            # Long: break above R1 with squeeze
            if close[i] > r1_aligned[i] and squeeze[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below S1 with squeeze
            elif close[i] < s1_aligned[i] and squeeze[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: return to pivot point or squeeze breaks
            if close[i] <= pp_aligned[i] or not squeeze[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: return to pivot point or squeeze breaks
            if close[i] >= pp_aligned[i] or not squeeze[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WeeklyPivot_Squeeze_Breakout"
timeframe = "1d"
leverage = 1.0