#!/usr/bin/env python3
"""
12h_Camarilla_R3S3_Reversal_1dTrend_Filter
Hypothesis: On 12h timeframe, price tends to reverse from 1d Camarilla R3/S3 levels when the 1d trend (EMA34) is aligned, providing mean-reversion opportunities in ranging conditions and continuation in strong trends. Uses 1d EMA34 trend filter to avoid counter-trend trades, targeting 15-30 trades/year to minimize fee drag.
"""

name = "12h_Camarilla_R3S3_Reversal_1dTrend_Filter"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for trend filter and Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    # 12h OHLCV
    close_12h = prices['close'].values
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    
    # --- 1d EMA34 for trend filter ---
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h = align_htf_to_ltf(prices, df_1d, ema34_1d)  # aligned to 12h
    
    # --- 1d Camarilla Pivot Levels (using previous day) ---
    # Calculate from previous day's OHLC
    prev_high = np.roll(df_1d['high'].values, 1)
    prev_low = np.roll(df_1d['low'].values, 1)
    prev_close = np.roll(df_1d['close'].values, 1)
    prev_high[0] = df_1d['high'].values[0]
    prev_low[0] = df_1d['low'].values[0]
    prev_close[0] = df_1d['close'].values[0]
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_val = prev_high - prev_low
    
    # Camarilla levels
    r3 = pivot + (range_val * 1.1 / 2.0)
    s3 = pivot - (range_val * 1.1 / 2.0)
    
    # Align Camarilla levels to 12h
    r3_12h = align_htf_to_ltf(prices, df_1d, r3)
    s3_12h = align_htf_to_ltf(prices, df_1d, s3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start after warmup period
    start_idx = 35  # for EMA34 calculation
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema34_12h[i]) or np.isnan(r3_12h[i]) or np.isnan(s3_12h[i])):
            if position != 0:
                # Simple stoploss: 2.5% adverse move
                adverse_pct = 0.025
                if position == 1 and close_12h[i] <= entry_price * (1 - adverse_pct):
                    signals[i] = 0.0
                    position = 0
                elif position == -1 and close_12h[i] >= entry_price * (1 + adverse_pct):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25 if position == 1 else -0.25
            continue
        
        # Trend filter: price above EMA34 = uptrend, below = downtrend
        is_uptrend = close_12h[i] > ema34_12h[i]
        is_downtrend = close_12h[i] < ema34_12h[i]
        
        if position == 0:
            # Look for mean-reversion entries at Camarilla levels
            # Long when price touches S3 in uptrend or ranging market
            if close_12h[i] <= s3_12h[i] * 1.001:  # slight buffer for touching
                if is_uptrend or not is_downtrend:  # allow in uptrend or ranging
                    signals[i] = 0.25
                    position = 1
                    entry_price = close_12h[i]
            # Short when price touches R3 in downtrend or ranging market
            elif close_12h[i] >= r3_12h[i] * 0.999:  # slight buffer for touching
                if is_downtrend or not is_uptrend:  # allow in downtrend or ranging
                    signals[i] = -0.25
                    position = -1
                    entry_price = close_12h[i]
        else:
            # Manage existing position
            if position == 1:
                # Long position: exit when price reaches pivot or shows weakness
                if close_12h[i] >= pivot[i] * 0.999:  # reached pivot level
                    signals[i] = 0.0
                    position = 0
                # Stoploss: 2.5% adverse move
                elif close_12h[i] <= entry_price * (1 - 0.025):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Short position: exit when price reaches pivot or shows weakness
                if close_12h[i] <= pivot[i] * 1.001:  # reached pivot level
                    signals[i] = 0.0
                    position = 0
                # Stoploss: 2.5% adverse move
                elif close_12h[i] >= entry_price * (1 + 0.025):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals