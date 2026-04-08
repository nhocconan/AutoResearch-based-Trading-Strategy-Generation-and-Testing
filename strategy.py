#!/usr/bin/env python3
# 6h_weekly_pivot_breakout_1d_trend_v1
# Hypothesis: On 6h timeframe, use weekly pivot levels (from Monday open) with daily trend filter.
# Long when price breaks above weekly R1 with daily uptrend and volume confirmation.
# Short when price breaks below weekly S1 with daily downtrend and volume confirmation.
# Weekly pivots calculated from prior week's (Monday-Friday) OHLC.
# Target: 15-25 trades/year to minimize fee decay while capturing institutional breakouts.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_htf

name = "6h_weekly_pivot_breakout_1d_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using prior week's data)
    # Week runs Monday to Friday; we use previous week's data to avoid look-ahead
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot and support/resistance levels
    pivot_1w = (high_1w + low_1w + close_1w) / 3
    range_1w = high_1w - low_1w
    
    # Standard pivot levels (S1, R1 are primary)
    S1_1w = 2 * pivot_1w - high_1w
    R1_1w = 2 * pivot_1w - low_1w
    
    # Align weekly levels to 6h timeframe
    S1_6h = align_ltf_to_htf(prices, df_1w, S1_1w)
    R1_6h = align_ltf_to_htf(prices, df_1w, R1_1w)
    
    # Daily trend filter: EMA50 vs EMA200
    ema50 = pd.Series(close).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema200 = pd.Series(close).ewm(span=200, min_periods=200, adjust=False).mean().values
    uptrend = ema50 > ema200
    downtrend = ema50 < ema200
    
    # Volume confirmation: 24-period average (4 days of 6h bars)
    avg_volume = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 200
    
    for i in range(start_idx, n):
        # Skip if data not available
        if np.isnan(S1_6h[i]) or np.isnan(R1_6h[i]) or \
           np.isnan(ema50[i]) or np.isnan(ema200[i]) or \
           np.isnan(avg_volume[i]):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price breaks below S1 or opposite signal with volume
            if close[i] < S1_6h[i] or \
               (close[i] < R1_6h[i] and volume[i] > 2.0 * avg_volume[i] and downtrend[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above R1 or opposite signal with volume
            if close[i] > R1_6h[i] or \
               (close[i] > S1_6h[i] and volume[i] > 2.0 * avg_volume[i] and uptrend[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation: current volume > 1.5x average volume
            volume_ok = volume[i] > 1.5 * avg_volume[i]
            
            # Long entry: price breaks above R1 with volume and uptrend bias
            if close[i] > R1_6h[i] and volume_ok and uptrend[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below S1 with volume and downtrend bias
            elif close[i] < S1_6h[i] and volume_ok and downtrend[i]:
                position = -1
                signals[i] = -0.25
    
    return signals