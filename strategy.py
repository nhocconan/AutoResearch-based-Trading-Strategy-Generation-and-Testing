# 6h_Pivot_Reversal_12hTrend_Filter
# Hypothesis: Daily pivot reversals during trending 12h markets capture short-term mean reversion within larger trends.
# Works in bull/bear by using 12h trend direction (via EMA34) to filter pivot reversals (fade at R1/S1, breakout at R4/S4).
# Volume confirmation ensures institutional participation. Targets 15-30 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_pivots(high, low, close):
    """Calculate standard pivot points and support/resistance levels"""
    pivot = (high + low + close) / 3.0
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    r3 = high + 2 * (pivot - low)
    s3 = low - 2 * (high - pivot)
    r4 = 3 * pivot - 2 * low
    s4 = 3 * pivot - 2 * high
    return pivot, r1, r2, r3, r4, s1, s2, s3, s4

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter and pivot calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    # Calculate EMA34 on 12h for trend filter
    close_12h = df_12h['close'].values
    ema34_12h = pd.Series(close_12h).ewm(span=34, adjust=False).mean().values
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    # Calculate pivots on 12h using previous bar's data (standard pivot calculation)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    pivot_12h, r1_12h, r2_12h, r3_12h, r4_12h, s1_12h, s2_12h, s3_12h, s4_12h = calculate_pivots(
        high_12h[:-1], low_12h[:-1], close_12h[:-1]  # Use previous bar for today's pivot
    )
    # Align pivot levels to 6h timeframe
    pivot_12h_aligned = align_htf_to_ltf(prices, df_12h, pivot_12h)
    r1_12h_aligned = align_htf_to_ltf(prices, df_12h, r1_12h)
    r2_12h_aligned = align_htf_to_ltf(prices, df_12h, r2_12h)
    r3_12h_aligned = align_htf_to_ltf(prices, df_12h, r3_12h)
    r4_12h_aligned = align_htf_to_ltf(prices, df_12h, r4_12h)
    s1_12h_aligned = align_htf_to_ltf(prices, df_12h, s1_12h)
    s2_12h_aligned = align_htf_to_ltf(prices, df_12h, s2_12h)
    s3_12h_aligned = align_htf_to_ltf(prices, df_12h, s3_12h)
    s4_12h_aligned = align_htf_to_ltf(prices, df_12h, s4_12h)
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma = np.zeros_like(volume)
    for i in range(len(volume)):
        if i < 20:
            vol_ma[i] = np.mean(volume[max(0, i-19):i+1]) if i >= 0 else volume[i]
        else:
            vol_ma[i] = np.mean(volume[i-19:i+1])
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Warmup for EMA34
    
    for i in range(start_idx, n):
        if np.isnan(ema34_12h_aligned[i]) or np.isnan(pivot_12h_aligned[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        # Trend filter: price relative to 12h EMA34
        uptrend = close[i] > ema34_12h_aligned[i]
        downtrend = close[i] < ema34_12h_aligned[i]
        vol_ok = vol_spike[i]
        
        if position == 0:
            # Long setup: 
            # 1. In uptrend (price > EMA34) AND price crosses above S1 (bounce from support)
            # 2. OR in downtrend (price < EMA34) AND price breaks above R4 (strong breakout)
            if (uptrend and close[i] > s1_12h_aligned[i] and close[i-1] <= s1_12h_aligned[i]) or \
               (downtrend and close[i] > r4_12h_aligned[i] and close[i-1] <= r4_12h_aligned[i]):
                if vol_ok:
                    signals[i] = 0.25
                    position = 1
            
            # Short setup:
            # 1. In downtrend (price < EMA34) AND price crosses below R1 (rejection at resistance)
            # 2. OR in uptrend (price > EMA34) AND price breaks below S4 (strong breakdown)
            elif (downtrend and close[i] < r1_12h_aligned[i] and close[i-1] >= r1_12h_aligned[i]) or \
                 (uptrend and close[i] < s4_12h_aligned[i] and close[i-1] >= s4_12h_aligned[i]):
                if vol_ok:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit long: price crosses below pivot (mean reversion) or trend breaks down
            if close[i] < pivot_12h_aligned[i] or (downtrend and close[i] < ema34_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above pivot (mean reversion) or trend breaks up
            if close[i] > pivot_12h_aligned[i] or (uptrend and close[i] > ema34_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Pivot_Reversal_12hTrend_Filter"
timeframe = "6h"
leverage = 1.0