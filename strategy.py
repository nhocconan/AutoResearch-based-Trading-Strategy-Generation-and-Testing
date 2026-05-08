#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_WeeklyPivot_Momentum_Filter"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data once for pivot points and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Get daily data for momentum and volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate weekly pivot points (standard formula)
    # PP = (H + L + C) / 3
    # R1 = 2*PP - L
    # S1 = 2*PP - H
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Previous week's values
    prev_close_1w = np.roll(close_1w, 1)
    prev_high_1w = np.roll(high_1w, 1)
    prev_low_1w = np.roll(low_1w, 1)
    prev_close_1w[0] = close_1w[0]
    prev_high_1w[0] = high_1w[0]
    prev_low_1w[0] = low_1w[0]
    
    # Weekly pivot levels
    PP = (prev_high_1w + prev_low_1w + prev_close_1w) / 3
    R1 = 2 * PP - prev_low_1w
    S1 = 2 * PP - prev_high_1w
    
    # Align weekly pivot levels to 6h timeframe
    PP_aligned = align_htf_to_ltf(prices, df_1w, PP)
    R1_aligned = align_htf_to_ltf(prices, df_1w, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1w, S1)
    
    # Daily momentum: RSI(14) > 50 for long, < 50 for short
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi_ma = pd.Series(rsi).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align RSI to 6h timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi_ma)
    
    # Daily volume confirmation: volume > 1.3 * 20-day average
    volume_1d = df_1d['volume'].values
    vol_ma20d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_confirmed = volume_1d > (vol_ma20d * 1.3)
    vol_confirmed_aligned = align_htf_to_ltf(prices, df_1d, vol_confirmed)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup for all indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(PP_aligned[i]) or np.isnan(R1_aligned[i]) or 
            np.isnan(S1_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(vol_confirmed_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: price above weekly pivot, RSI > 50, volume confirmed
            long_cond = (close[i] > PP_aligned[i] and 
                        rsi_aligned[i] > 50 and 
                        vol_confirmed_aligned[i])
            
            # Short entry: price below weekly pivot, RSI < 50, volume confirmed
            short_cond = (close[i] < PP_aligned[i] and 
                         rsi_aligned[i] < 50 and 
                         vol_confirmed_aligned[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below S1 (strong support break)
            if close[i] < S1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above R1 (strong resistance break)
            if close[i] > R1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Weekly pivot points act as significant support/resistance levels.
# In ranging markets, price tends to revert to the pivot (PP).
# In trending markets, price breaks through R1/S1 with momentum.
# RSI(14) filter ensures we only trade in the direction of momentum.
# Volume confirmation (1.3x 20-day average) ensures institutional participation.
# Works in both bull (breakouts) and bear (mean reversion) markets.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.