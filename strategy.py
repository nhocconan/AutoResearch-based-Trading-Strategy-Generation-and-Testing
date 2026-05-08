#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Pivot_Trend_Switch_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for pivots and 12h data for trend
    df_1d = get_htf_data(prices, '1d')
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_1d) < 30 or len(df_12h) < 30:
        return np.zeros(n)
    
    # === 1d Pivot Points (standard calculation) ===
    prev_high = np.roll(df_1d['high'].values, 1)
    prev_low = np.roll(df_1d['low'].values, 1)
    prev_close = np.roll(df_1d['close'].values, 1)
    # First bar: use current values to avoid look-ahead
    prev_high[0] = df_1d['high'].values[0]
    prev_low[0] = df_1d['low'].values[0]
    prev_close[0] = df_1d['close'].values[0]
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_1d = prev_high - prev_low
    
    # Key levels: R1, S1, R2, S2
    r1 = pivot + (range_1d * 1.0 / 2.0)  # Standard R1
    s1 = pivot - (range_1d * 1.0 / 2.0)  # Standard S1
    r2 = pivot + range_1d                # Standard R2
    s2 = pivot - range_1d                # Standard S2
    
    # Align to 4h timeframe
    r1_4h = align_htf_to_ltf(prices, df_1d, r1)
    s1_4h = align_htf_to_ltf(prices, df_1d, s1)
    r2_4h = align_htf_to_ltf(prices, df_1d, r2)
    s2_4h = align_htf_to_ltf(prices, df_1d, s2)
    
    # === 12h EMA25 for trend filter ===
    close_12h = df_12h['close'].values
    ema25_12h = pd.Series(close_12h).ewm(span=25, adjust=False, min_periods=25).mean().values
    ema25_12h_4h = align_htf_to_ltf(prices, df_12h, ema25_12h)
    
    # === Volume confirmation: 20-period average ===
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # warmup for EMA25
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(r1_4h[i]) or np.isnan(s1_4h[i]) or np.isnan(r2_4h[i]) or np.isnan(s2_4h[i]) or
            np.isnan(ema25_12h_4h[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Determine trend: above/below 12h EMA25
            uptrend = close[i] > ema25_12h_4h[i]
            downtrend = close[i] < ema25_12h_4h[i]
            
            if uptrend:
                # In uptrend: look for pullback to S1 for long entry
                long_cond = (close[i] <= s1_4h[i] * 1.005 and  # Allow small buffer
                            volume[i] > vol_ma20[i])
                
                if long_cond:
                    signals[i] = 0.25
                    position = 1
            elif downtrend:
                # In downtrend: look for bounce to R1 for short entry
                short_cond = (close[i] >= r1_4h[i] * 0.995 and  # Allow small buffer
                             volume[i] > vol_ma20[i])
                
                if short_cond:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: either take profit at R1 or stop if trend breaks
            if close[i] >= r1_4h[i] * 0.995:  # Take profit near R1
                signals[i] = 0.0
                position = 0
            elif close[i] < ema25_12h_4h[i]:  # Trend breakdown
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: either take profit at S1 or stop if trend breaks
            if close[i] <= s1_4h[i] * 1.005:  # Take profit near S1
                signals[i] = 0.0
                position = 0
            elif close[i] > ema25_12h_4h[i]:  # Trend reversal
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Pivot-based trend-following strategy that buys pullbacks to S1 in uptrends
# and sells bounces to R1 in downtrends. Uses 12h EMA25 as trend filter and volume
# confirmation for institutional validation. Designed to work in both bull and bear
# markets by following the higher timeframe trend. Targets 50-150 trades over 4 years
# (12-37/year) with discrete sizing (0.25) to minimize fee drag. Works on BTC/ETH via
# institutional pivot levels that act as support/resistance in trending markets.