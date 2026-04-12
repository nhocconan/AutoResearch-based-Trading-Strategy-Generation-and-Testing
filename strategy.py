#!/usr/bin/env python3
"""
12h_1d_Camarilla_Breakout_Trend_v1
Hypothesis: On 12h timeframe, buy breakouts above daily Camarilla H3 with 1d uptrend filter and volume confirmation,
sell breakdowns below L3 with 1d downtrend and volume confirmation. Exit at H4/L4 levels.
Designed for low trade frequency (12-37/year) by requiring multiple confluence factors.
Works in bull/bear via 1d trend filter and mean-reversion exit at Camarilla levels.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_Camarilla_Breakout_Trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === DAILY CAMARILLA LEVELS ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Use previous day's close for Camarilla calculation (no look-ahead)
    close_prev = np.concatenate([[close_1d[0]], close_1d[:-1]])
    range_1d = high_1d - low_1d
    
    h3 = close_prev + (range_1d * 1.1 / 4)
    l3 = close_prev - (range_1d * 1.1 / 4)
    h4 = close_prev + (range_1d * 1.1)
    l4 = close_prev - (range_1d * 1.1)
    
    # === DAILY TREND FILTER: EMA25 > EMA50 for uptrend, < for downtrend ===
    close_series = pd.Series(close_1d)
    ema25 = close_series.ewm(span=25, adjust=False).values
    ema50 = close_series.ewm(span=50, adjust=False).values
    uptrend = ema25 > ema50
    downtrend = ema25 < ema50
    
    # === VOLUME CONFIRMATION: 20-period average ===
    vol_avg = np.zeros(n)
    vol_sum = 0.0
    vol_count = 0
    for i in range(n):
        vol_sum += volume[i]
        vol_count += 1
        if i >= 20:
            vol_sum -= volume[i-20]
            vol_count -= 1
        vol_avg[i] = vol_sum / vol_count if vol_count > 0 else 0.0
    
    # Align all data to 12h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    uptrend_aligned = align_htf_to_ltf(prices, df_1d, uptrend.astype(float))
    downtrend_aligned = align_htf_to_ltf(prices, df_1d, downtrend.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # start after warmup
        # Skip if indicators not available
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or 
            np.isnan(uptrend_aligned[i]) or np.isnan(downtrend_aligned[i]) or 
            vol_avg[i] == 0.0):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: at least 1.5x average
        vol_confirm = volume[i] > 1.5 * vol_avg[i]
        
        # Entry conditions
        long_setup = (close[i] > h3_aligned[i]) and vol_confirm and (uptrend_aligned[i] > 0.5)
        short_setup = (close[i] < l3_aligned[i]) and vol_confirm and (downtrend_aligned[i] > 0.5)
        
        # Exit conditions: mean reversion to H4/L4 levels
        exit_long = close[i] < l4_aligned[i]
        exit_short = close[i] > h4_aligned[i]
        
        if long_setup and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_setup and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals