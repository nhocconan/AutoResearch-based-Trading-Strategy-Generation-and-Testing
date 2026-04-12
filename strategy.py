#!/usr/bin/env python3
"""
6h_1d_weekly_pivot_reversion
Hypothesis: Price tends to revert to weekly pivot points (PP) on 6h timeframe.
- Weekly PP calculated from prior week's (H+L+C)/3
- Buy when price touches S1 (PP - (R1-S1)) with RSI < 40 and volume > 1.5x avg
- Sell when price touches R1 (PP + (R1-S1)) with RSI > 60 and volume > 1.5x avg
- Uses weekly pivot for multi-day mean reversion edge in both bull/bear markets
- Target: 15-30 trades/year (60-120 total over 4 years) to minimize fee drag
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_weekly_pivot_reversion"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Previous week's data for pivot (to avoid look-ahead)
    prev_high = np.roll(high_1w, 1)
    prev_low = np.roll(low_1w, 1)
    prev_close = np.roll(close_1w, 1)
    
    # Weekly pivot point and support/resistance levels
    pp = (prev_high + prev_low + prev_close) / 3.0
    range_ = prev_high - prev_low
    r1 = pp + range_
    s1 = pp - range_
    r2 = pp + 2 * range_
    s2 = pp - 2 * range_
    
    # Align weekly levels to 6h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1w, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    
    # RSI(14) for overbought/oversold
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean()
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(rsi[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long entry: price at S1 with oversold RSI and volume
        if (abs(close[i] - s1_aligned[i]) < 0.001 * close[i] and  # within 0.1% of S1
            rsi[i] < 40 and vol_confirm[i] and position != 1):
            position = 1
            signals[i] = 0.25
        # Short entry: price at R1 with overbought RSI and volume
        elif (abs(close[i] - r1_aligned[i]) < 0.001 * close[i] and  # within 0.1% of R1
              rsi[i] > 60 and vol_confirm[i] and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: price moves back to pivot or opposite S1/R1
        elif position == 1 and (close[i] > pp_aligned[i] or close[i] < s1_aligned[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] < pp_aligned[i] or close[i] > r1_aligned[i]):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals