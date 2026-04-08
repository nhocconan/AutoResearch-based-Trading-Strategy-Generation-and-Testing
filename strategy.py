#!/usr/bin/env python3
# 6h_weekly_pivot_reversion_v1
# Hypothesis: Mean reversion at weekly pivot levels on 6h timeframe.
# Weekly pivot levels act as strong support/resistance due to institutional order flow.
# Price tends to reverse from weekly S1/R1 and S2/R2 levels, especially when
# combined with 6h RSI extremes and volume confirmation.
# Works in both bull and bear markets as pivot levels are recalculated weekly.
# Target: 15-25 trades/year (~60-100 total over 4 years) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_weekly_pivot_reversion_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using previous week's OHLC)
    # Pivot = (H + L + C) / 3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    pivot = (high_1w + low_1w + close_1w) / 3.0
    r1 = 2 * pivot - low_1w
    s1 = 2 * pivot - high_1w
    r2 = pivot + (high_1w - low_1w)
    s2 = pivot - (high_1w - low_1w)
    
    # Align weekly pivot levels to 6h timeframe (with 1-week delay for completion)
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    
    # Calculate 6h RSI for overbought/oversold signals
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing (equivalent to RMA)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(rsi[i]) or np.isnan(pivot_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price reaches S1 or RSI becomes overbought
            if (close[i] <= s1_aligned[i]) or (rsi[i] >= 70):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price reaches R1 or RSI becomes oversold
            if (close[i] >= r1_aligned[i]) or (rsi[i] <= 30):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price at S2 with RSI oversold and volume confirmation
            if (close[i] <= s2_aligned[i] * 1.002) and (rsi[i] < 30) and vol_filter[i]:
                # Additional confirmation: price should be above S1 to avoid falling knife
                if close[i] > s1_aligned[i]:
                    position = 1
                    signals[i] = 0.25
            # Short entry: price at R2 with RSI overbought and volume confirmation
            elif (close[i] >= r2_aligned[i] * 0.998) and (rsi[i] > 70) and vol_filter[i]:
                # Additional confirmation: price should be below R1 to avoid chasing
                if close[i] < r1_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals