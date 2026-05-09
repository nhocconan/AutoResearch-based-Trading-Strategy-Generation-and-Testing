#!/usr/bin/env python3
name = "6H_WeeklyPivot_MeanReversion_TrendFilter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot points
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate weekly pivot and support/resistance levels
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot point (standard calculation)
    pivot_1w = (high_1w + low_1w + close_1w) / 3
    range_1w = high_1w - low_1w
    
    # Weekly support and resistance levels
    r1_1w = 2 * pivot_1w - low_1w
    s1_1w = 2 * pivot_1w - high_1w
    r2_1w = pivot_1w + range_1w
    s2_1w = pivot_1w - range_1w
    
    # Align weekly levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2_1w)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2_1w)
    
    # Daily EMA50 for trend filter (to avoid mean reversion in strong trends)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume filter: current volume > 1.5x 20-period average (avoid low-volume noise)
    volume_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (volume_avg * 1.5)
    
    # RSI(14) for overbought/oversold confirmation
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or np.isnan(ema50_aligned[i]) or 
            np.isnan(rsi_values[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price near S1 (support) AND above weekly pivot (bullish bias) AND not oversold
            if (close[i] <= s1_aligned[i] * 1.02 and  # within 2% of S1
                close[i] > pivot_aligned[i] and       # above weekly pivot
                rsi_values[i] < 40 and              # not overbought
                volume_filter[i]):                  # volume confirmation
                signals[i] = 0.25
                position = 1
            # Enter short: price near R1 (resistance) AND below weekly pivot (bearish bias) AND not oversold
            elif (close[i] >= r1_aligned[i] * 0.98 and  # within 2% of R1
                  close[i] < pivot_aligned[i] and       # below weekly pivot
                  rsi_values[i] > 60 and              # not oversold
                  volume_filter[i]):                  # volume confirmation
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price reaches pivot (profit target) OR RSI overbought OR breaks below S1
            if (close[i] >= pivot_aligned[i] or    # reached pivot (target)
                rsi_values[i] > 70 or              # overbought
                close[i] < s1_aligned[i] * 0.98):  # broke below support
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price reaches pivot (profit target) OR RSI oversold OR breaks above R1
            if (close[i] <= pivot_aligned[i] or    # reached pivot (target)
                rsi_values[i] < 30 or              # oversold
                close[i] > r1_aligned[i] * 1.02):  # broke above resistance
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals