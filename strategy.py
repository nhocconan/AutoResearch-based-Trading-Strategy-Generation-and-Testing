#!/usr/bin/env python3
"""
6h Weekly Pivot + Volume Confirmation
Hypothesis: Weekly pivot levels act as strong support/resistance. Price tends to respect these levels,
especially when combined with volume confirmation. Long when price bounces from S1/S2 with bullish volume,
short when price rejects R1/R2 with bearish volume. Works in both bull and bear markets as price
respects key levels regardless of trend. Target: 75-150 total trades over 4 years (19-38/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "14367_6h_weekly_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data for pivot calculation (once before loop)
    df_w = get_htf_data(prices, '1w')
    high_w = df_w['high'].values
    low_w = df_w['low'].values
    close_w = df_w['close'].values
    
    # Calculate weekly pivot points (using previous week's data)
    # Pivot Point (P) = (High + Low + Close) / 3
    # Support 1 (S1) = (2 * P) - High
    # Support 2 (S2) = P - (High - Low)
    # Resistance 1 (R1) = (2 * P) - Low
    # Resistance 2 (R2) = P + (High - Low)
    pp = (high_w + low_w + close_w) / 3
    s1 = (2 * pp) - high_w
    s2 = pp - (high_w - low_w)
    r1 = (2 * pp) - low_w
    r2 = pp + (high_w - low_w)
    
    # Align weekly pivot levels to 6h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_w, pp)
    s1_aligned = align_htf_to_ltf(prices, df_w, s1)
    s2_aligned = align_htf_to_ltf(prices, df_w, s2)
    r1_aligned = align_htf_to_ltf(prices, df_w, r1)
    r2_aligned = align_htf_to_ltf(prices, df_w, r2)
    
    # 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume filter: require volume above average to confirm interest
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (0.7 * vol_ma)  # At least 70% of average volume
    
    # ATR for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = 20  # Need enough data for volume MA
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(pp_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(s2_aligned[i]) or
            np.isnan(r1_aligned[i]) or np.isnan(r2_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits
        if position == 1:  # long position
            # Exit: price breaks below S1 OR stoploss
            if (close[i] <= s1_aligned[i] or close[i] <= entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above R1 OR stoploss
            if (close[i] >= r1_aligned[i] or close[i] >= entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: price near pivot levels + volume confirmation
            # Long: price near S1 or S2 with bullish rejection (close > open) and volume
            near_s1 = abs(close[i] - s1_aligned[i]) <= (0.5 * atr[i])
            near_s2 = abs(close[i] - s2_aligned[i]) <= (0.5 * atr[i])
            bullish_candle = close[i] > prices['open'].values[i]
            
            # Short: price near R1 or R2 with bearish rejection (close < open) and volume
            near_r1 = abs(close[i] - r1_aligned[i]) <= (0.5 * atr[i])
            near_r2 = abs(close[i] - r2_aligned[i]) <= (0.5 * atr[i])
            bearish_candle = close[i] < prices['open'].values[i]
            
            long_setup = ((near_s1 or near_s2) and bullish_candle and vol_filter[i])
            short_setup = ((near_r1 or near_r2) and bearish_candle and vol_filter[i])
            
            if long_setup:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif short_setup:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals