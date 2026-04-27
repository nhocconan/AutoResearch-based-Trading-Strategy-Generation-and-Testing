#!/usr/bin/env python3
"""
12h Camarilla Pivot Reversion with 1d Trend Filter and Volume Spike.
Long when price touches S1/S2 and closes back above, with 1d EMA34 uptrend and volume spike.
Short when price touches R1/R2 and closes back below, with 1d EMA34 downtrend and volume spike.
Exit on touch of opposite H/L level or close beyond 3/4 levels.
Designed for 15-35 trades/year per symbol with mean-reversion edge in ranging markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given period"""
    range_val = high - low
    if range_val <= 0:
        return close, close, close, close, close, close, close, close
    c = close
    h = high
    l = low
    # Resistance levels
    r4 = c + ((h - l) * 1.5000)
    r3 = c + ((h - l) * 1.2500)
    r2 = c + ((h - l) * 1.1666)
    r1 = c + ((h - l) * 1.0833)
    # Support levels
    s1 = c - ((h - l) * 1.0833)
    s2 = c - ((h - l) * 1.1666)
    s3 = c - ((h - l) * 1.2500)
    s4 = c - ((h - l) * 1.5000)
    return r4, r3, r2, r1, s1, s2, s3, s4

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter (EMA34)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate EMA34 on 1d close
    close_1d = df_1d['close'].values
    ema_34_1d = np.empty_like(close_1d, dtype=np.float64)
    ema_34_1d.fill(np.nan)
    if len(close_1d) >= 34:
        # Use pandas EMA for accuracy and proper handling
        ema_series = pd.Series(close_1d).ewm(span=34, adjust=False).values
        ema_34_1d[:] = ema_series
    
    # Align EMA34 to 12h timeframe
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from 12h OHLC (use lookback of 1 period)
    r4 = np.full(n, np.nan)
    r3 = np.full(n, np.nan)
    r2 = np.full(n, np.nan)
    r1 = np.full(n, np.nan)
    s1 = np.full(n, np.nan)
    s2 = np.full(n, np.nan)
    s3 = np.full(n, np.nan)
    s4 = np.full(n, np.nan)
    
    # Calculate for each bar using previous bar's HLC (standard pivot calculation)
    for i in range(1, n):
        r4[i], r3[i], r2[i], r1[i], s1[i], s2[i], s3[i], s4[i] = calculate_camarilla(
            high[i-1], low[i-1], close[i-1]
        )
    
    # Volume filter: volume > 2.0x average (to avoid false signals)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need EMA34 (34) and volume MA (20)
    start_idx = max(34, 19) + 1  # +1 because we use previous bar for pivots
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_aligned[i]) or np.isnan(r1[i]) or np.isnan(s1[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Current price and volume
        price_now = close[i]
        vol_now = volume[i]
        
        # Current levels
        r1_val = r1[i]
        r2_val = r2[i]
        s1_val = s1[i]
        s2_val = s2[i]
        r3_val = r3[i]
        r4_val = r4[i]
        s3_val = s3[i]
        s4_val = s4[i]
        ema_34_val = ema_34_aligned[i]
        
        # Volume filter: volume > 2.0x average
        vol_filter = vol_now > 2.0 * vol_ma_20[i]
        
        if position == 0:
            # Long setup: price touches S1/S2 and closes back above, with uptrend and volume spike
            if ((price_now <= s1_val * 1.001 and close[i] > s1_val) or 
                (price_now <= s2_val * 1.001 and close[i] > s2_val)) and \
               ema_34_val > close_1d[-1] if len(close_1d) > 0 else False and vol_filter:
                # Simplified trend check: current price above yesterday's close (proxy for uptrend)
                if i >= 1 and close[i] > close[i-1]:
                    signals[i] = size
                    position = 1
            # Short setup: price touches R1/R2 and closes back below, with downtrend and volume spike
            elif ((price_now >= r1_val * 0.999 and close[i] < r1_val) or 
                  (price_now >= r2_val * 0.999 and close[i] < r2_val)) and \
                 ema_34_val < close_1d[-1] if len(close_1d) > 0 else False and vol_filter:
                # Simplified trend check: current price below yesterday's close (proxy for downtrend)
                if i >= 1 and close[i] < close[i-1]:
                    signals[i] = -size
                    position = -1
        elif position == 1:
            # Exit long: price touches R1 or closes below S1
            if price_now >= r1_val * 0.999 or close[i] < s1_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price touches S1 or closes above R1
            if price_now <= s1_val * 1.001 or close[i] > r1_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Camarilla_Pivot_Reversion_Trend_Filter"
timeframe = "12h"
leverage = 1.0