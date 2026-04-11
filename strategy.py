#!/usr/bin/env python3
# 1h_1d_4h_camarilla_pivot_volume_v1
# Strategy: 1h Camarilla Pivot reversal with volume confirmation and 4h/1d trend filter
# Timeframe: 1h
# Leverage: 1.0
# Hypothesis: In ranging markets (common in 2025 BTC/ETH), price reverts from Camarilla H3/L3 levels.
# In trending markets, breaks of H4/L4 with volume and 4h/1d alignment capture moves.
# Volume confirms legitimacy. Uses 4h/1d for direction, 1h for entry. Targets 15-35 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_1d_4h_camarilla_pivot_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data for Camarilla calculation (using prior day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    # H4 = C + (H-L)*1.1/2, L4 = C - (H-L)*1.1/2
    # H3 = C + (H-L)*1.1/4, L3 = C - (H-L)*1.1/4
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Avoid look-ahead: use previous day's data only
    rng = prev_high - prev_low
    h4 = prev_close + rng * 1.1 / 2.0
    l4 = prev_close - rng * 1.1 / 2.0
    h3 = prev_close + rng * 1.1 / 4.0
    l3 = prev_close - rng * 1.1 / 4.0
    
    # Align daily levels to 1h timeframe (hold until next day)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    
    # 4h EMA(20) for trend filter
    close_4h = df_4h['close'].values
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # Volume confirmation: current volume > 1.5x 24-period average
    vol_series = pd.Series(volume)
    vol_avg_24 = vol_series.rolling(window=24, min_periods=24).mean().values
    vol_confirm = volume > (1.5 * vol_avg_24)
    
    # Session filter: 08:00-20:00 UTC
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(24, n):
        # Skip if any required data is invalid or outside session
        if (np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or 
            np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(ema_20_4h_aligned[i]) or not in_session[i]):
            # Flatten position if invalid
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Trend filter from 4h EMA
        uptrend_4h = close[i] > ema_20_4h_aligned[i]
        downtrend_4h = close[i] < ema_20_4h_aligned[i]
        
        # Entry logic
        if position == 0:
            # Long setup: price rejects L3/H3 with volume in uptrend, or breaks L4 with volume
            long_break = close[i] > l4_aligned[i] and vol_confirm[i] and uptrend_4h
            long_reject = close[i] > l3_aligned[i] and close[i] < l3_aligned[i] + (h3_aligned[i]-l3_aligned[i])*0.1 and vol_confirm[i]
            
            # Short setup: price rejects H3/L3 with volume in downtrend, or breaks H4 with volume
            short_break = close[i] < h4_aligned[i] and vol_confirm[i] and downtrend_4h
            short_reject = close[i] < h3_aligned[i] and close[i] > h3_aligned[i] - (h3_aligned[i]-l3_aligned[i])*0.1 and vol_confirm[i]
            
            if long_break or long_reject:
                position = 1
                signals[i] = 0.20
            elif short_break or short_reject:
                position = -1
                signals[i] = -0.20
            else:
                signals[i] = 0.0
                
        elif position == 1:
            # Long exit: price reaches H3 or breaks below L4 with volume
            if close[i] >= h3_aligned[i] or (close[i] < l4_aligned[i] and vol_confirm[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:
            # Short exit: price reaches L3 or breaks above H4 with volume
            if close[i] <= l3_aligned[i] or (close[i] > h4_aligned[i] and vol_confirm[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
    
    return signals