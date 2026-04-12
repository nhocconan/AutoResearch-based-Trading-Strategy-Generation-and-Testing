#!/usr/bin/env python3
"""
6h_1w_1d_Camarilla_Breakout_v1
Hypothesis: On 6h timeframe, use weekly Camarilla pivot levels (from 1w data) to identify
strong support/resistance zones. Go long when price breaks above weekly R3 with 1d
uptrend filter; short when breaks below weekly S3 with 1d downtrend filter. Exit at
opposite H4/L4 levels. Uses volume confirmation to avoid false breakouts.
Designed for low trade frequency (15-30/year) by requiring weekly level confluence
and trend alignment. Works in bull/bear via 1d trend filter and mean-reversion exit
at H4/L4 levels.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_1d_Camarilla_Breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === WEEKLY CAMARILLA LEVELS ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot points
    pivot_1w = (high_1w + low_1w + close_1w) / 3
    range_1w = high_1w - low_1w
    
    # Camarilla levels (weekly)
    r3_1w = close_1w + (1.1 * range_1w * 1.0 / 4.0)  # Close + 1.1*range/4
    s3_1w = close_1w - (1.1 * range_1w * 1.0 / 4.0)  # Close - 1.1*range/4
    r4_1w = close_1w + (1.1 * range_1w * 1.5 / 4.0)  # Close + 1.1*range*1.5/4
    s4_1w = close_1w - (1.1 * range_1w * 1.5 / 4.0)  # Close - 1.1*range*1.5/4
    
    # === DAILY EMA(20) FOR TREND FILTER ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    if len(close_1d) >= 20:
        ema_20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    else:
        ema_20_1d = np.full_like(close_1d, np.nan)
    
    # Align data to 6h timeframe
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    r4_1w_aligned = align_htf_to_ltf(prices, df_1w, r4_1w)
    s4_1w_aligned = align_htf_to_ltf(prices, df_1w, s4_1w)
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    # Volume average (24-period for 6h = ~6 days) for confirmation
    vol_avg = np.zeros(n)
    vol_sum = 0.0
    vol_count = 0
    for i in range(n):
        vol_sum += volume[i]
        vol_count += 1
        if i >= 24:
            vol_sum -= volume[i-24]
            vol_count -= 1
        if vol_count > 0:
            vol_avg[i] = vol_sum / vol_count
        else:
            vol_avg[i] = 0.0
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # start after warmup
        # Skip if indicators not available
        if (np.isnan(r3_1w_aligned[i]) or np.isnan(s3_1w_aligned[i]) or 
            np.isnan(ema_20_1d_aligned[i]) or vol_avg[i] == 0.0):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: at least 1.8x average
        vol_confirm = volume[i] > 1.8 * vol_avg[i]
        
        # Trend filter: price above/below 1d EMA(20)
        price_above_ema = close[i] > ema_20_1d_aligned[i]
        price_below_ema = close[i] < ema_20_1d_aligned[i]
        
        # Entry conditions: breakout of weekly R3/S3 with trend alignment
        long_setup = (close[i] > r3_1w_aligned[i]) and vol_confirm and price_above_ema
        short_setup = (close[i] < s3_1w_aligned[i]) and vol_confirm and price_below_ema
        
        # Exit conditions: mean reversion to opposite H4/L4 levels
        exit_long = close[i] < s4_1w_aligned[i]
        exit_short = close[i] > r4_1w_aligned[i]
        
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