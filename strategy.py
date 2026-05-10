#!/usr/bin/env python3
"""
1H_Camarilla_R1_S1_Breakout_4hTrend
Hypothesis: Camarilla R1/S1 breakout with 4h trend filter and volume confirmation.
Long: price breaks above R1, 4h trend up (EMA50 > EMA200), volume > 2x average.
Short: price breaks below S1, 4h trend down (EMA50 < EMA200), volume > 2x average.
Exit: price reverts to Camarilla Pivot point.
Uses 4h for trend direction and 1h for entry timing.
Target: 15-30 trades/year per symbol. Works in bull/bear by following 4h trend.
"""

name = "1H_Camarilla_R1_S1_Breakout_4hTrend"
timeframe = "1h"
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
    
    # Calculate Camarilla levels from previous day
    # Using daily OHLC from previous day for intraday levels
    # We'll use 24h lookback to get previous day's OHLC
    lookback = 24  # 24 hours for previous day
    if n < lookback + 20:
        return np.zeros(n)
    
    # Calculate previous day's high, low, close
    # For each bar, use the high/low/close from 24 bars ago
    ph = np.maximum.accumulate(high)
    pl = np.minimum.accumulate(low)
    # Shift by 24 to get previous day's values
    ph_prev = np.roll(ph, lookback)
    pl_prev = np.roll(pl, lookback)
    pc_prev = np.roll(close, lookback)
    
    # Set first lookback values to NaN
    ph_prev[:lookback] = np.nan
    pl_prev[:lookback] = np.nan
    pc_prev[:lookback] = np.nan
    
    # Camarilla levels
    range_prev = ph_prev - pl_prev
    # Avoid division by zero
    range_prev = np.where(range_prev == 0, 1e-10, range_prev)
    
    pivot = (ph_prev + pl_prev + pc_prev) / 3
    r1 = pc_prev + (range_prev * 1.1 / 12)
    s1 = pc_prev - (range_prev * 1.1 / 12)
    r4 = pc_prev + (range_prev * 1.1 / 2)
    s4 = pc_prev - (range_prev * 1.1 / 2)
    
    # 4h trend: EMA50 > EMA200 for uptrend, EMA50 < EMA200 for downtrend
    close_s = pd.Series(close)
    ema50 = close_s.ewm(span=50, adjust=False, min_periods=50).values
    ema200 = close_s.ewm(span=200, adjust=False, min_periods=200).values
    
    # Volume average (20-period)
    volume_s = pd.Series(volume)
    vol_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = max(lookback, 200) + 1
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50[i]) or np.isnan(ema200[i]) or np.isnan(vol_ma[i]) or
            np.isnan(r1[i]) or np.isnan(s1[i]) or np.isnan(pivot[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_confirm = vol_ratio > 2.0
        
        # 4h trend direction
        trend_up = ema50[i] > ema200[i]
        trend_down = ema50[i] < ema200[i]
        
        if position == 0:
            # Enter long: break above R1 + 4h uptrend + volume
            if close[i] > r1[i] and trend_up and volume_confirm:
                signals[i] = 0.20
                position = 1
            # Enter short: break below S1 + 4h downtrend + volume
            elif close[i] < s1[i] and trend_down and volume_confirm:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit when price returns to pivot (mean reversion)
            if close[i] <= pivot[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit when price returns to pivot
            if close[i] >= pivot[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals