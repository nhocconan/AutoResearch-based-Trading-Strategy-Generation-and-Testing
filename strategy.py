#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Daily_Pivot_Pullback_Trend_Filter_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once for pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # === Previous day's pivot points (HLC/3) ===
    prev_high_1d = np.roll(df_1d['high'].values, 1)
    prev_low_1d = np.roll(df_1d['low'].values, 1)
    prev_close_1d = np.roll(df_1d['close'].values, 1)
    prev_high_1d[0] = df_1d['high'].values[0]
    prev_low_1d[0] = df_1d['low'].values[0]
    prev_close_1d[0] = df_1d['close'].values[0]
    
    pivot = (prev_high_1d + prev_low_1d + prev_close_1d) / 3.0
    range_1d = prev_high_1d - prev_low_1d
    
    # Pivot support/resistance levels
    r1 = pivot + (range_1d * 1.1 / 12)
    s1 = pivot - (range_1d * 1.1 / 12)
    r2 = pivot + (range_1d * 1.1 / 6)
    s2 = pivot - (range_1d * 1.1 / 6)
    
    # Align pivot levels to 4h timeframe
    r1_4h = align_htf_to_ltf(prices, df_1d, r1)
    s1_4h = align_htf_to_ltf(prices, df_1d, s1)
    r2_4h = align_htf_to_ltf(prices, df_1d, r2)
    s2_4h = align_htf_to_ltf(prices, df_1d, s2)
    
    # === 4h EMA34 for trend filter (from experiment notes) ===
    ema34_4h = pd.Series(close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # === 4h Volume filter: current volume > 20-period average ===
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # warmup for EMA34
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(r1_4h[i]) or np.isnan(s1_4h[i]) or np.isnan(r2_4h[i]) or np.isnan(s2_4h[i]) or
            np.isnan(ema34_4h[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Pullback to S1 in uptrend (long)
            long_cond = (close[i] > ema34_4h[i] and  # Uptrend filter
                        close[i] <= s1_4h[i] * 1.01 and  # Near S1 (allow 1% tolerance)
                        close[i] >= s1_4h[i] * 0.99 and
                        volume[i] > vol_ma20[i])  # Volume confirmation
            
            # Pullback to R1 in downtrend (short)
            short_cond = (close[i] < ema34_4h[i] and  # Downtrend filter
                         close[i] >= r1_4h[i] * 0.99 and  # Near R1 (allow 1% tolerance)
                         close[i] <= r1_4h[i] * 1.01 and
                         volume[i] > vol_ma20[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price reaches R1 or trend reversal
            if close[i] >= r1_4h[i] or close[i] < ema34_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reaches S1 or trend reversal
            if close[i] <= s1_4h[i] or close[i] > ema34_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Pullback to daily pivot support/resistance levels with trend filter and volume confirmation.
# In uptrends (price > EMA34), look for long entries near S1; in downtrends, look for short entries near R1.
# Exits at opposite pivot level or trend reversal. Uses 4h timeframe for better trade frequency control.
# Designed to work in both bull (trend following pullbacks) and bear (mean reversion at pivots) markets.
# Targets 50-150 trades over 4 years (12-37/year) to minimize fee drag. Uses discrete sizing (0.25) to reduce churn.