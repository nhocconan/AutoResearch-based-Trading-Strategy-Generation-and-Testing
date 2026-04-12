#!/usr/bin/env python3
"""
4h_1d_Camarilla_Breakout_Signal_v2
Hypothesis: On 4h timeframe, long when price breaks above daily Camarilla H4 level with volume confirmation and 1d trend filter; short when breaks below L4 level with volume confirmation and 1d trend filter. Exit at opposite level (L4 for longs, H4 for shorts). Uses volume confirmation to avoid false breakouts and 1d EMA(50) for trend filter. Designed for low trade frequency (20-40/year) by requiring Camarilla level confluence and trend alignment. Works in bull/bear via 1d trend filter and mean-reversion exit at Camarilla levels.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_Camarilla_Breakout_Signal_v2"
timeframe = "4h"
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
    
    # Daily pivot and range
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Camarilla levels (H4/L4 for breakouts)
    h4_1d = close_1d + range_1d * 1.1 / 2
    l4_1d = close_1d - range_1d * 1.1 / 2
    
    # === 1D EMA(50) FOR TREND FILTER ===
    if len(close_1d) >= 50:
        ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    else:
        ema_50_1d = np.full_like(close_1d, np.nan)
    
    # Align data to 4h timeframe
    h4_1d_aligned = align_htf_to_ltf(prices, df_1d, h4_1d)
    l4_1d_aligned = align_htf_to_ltf(prices, df_1d, l4_1d)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume average (20-period for 4h = ~1 day) for confirmation
    vol_avg = np.zeros(n)
    vol_sum = 0.0
    vol_count = 0
    for i in range(n):
        vol_sum += volume[i]
        vol_count += 1
        if i >= 20:
            vol_sum -= volume[i-20]
            vol_count -= 1
        if vol_count > 0:
            vol_avg[i] = vol_sum / vol_count
        else:
            vol_avg[i] = 0.0
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # start after warmup
        # Skip if indicators not available
        if (np.isnan(h4_1d_aligned[i]) or np.isnan(l4_1d_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or vol_avg[i] == 0.0):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: at least 1.5x average
        vol_confirm = volume[i] > 1.5 * vol_avg[i]
        
        # Trend filter: price above/below 1d EMA(50)
        price_above_ema = close[i] > ema_50_1d_aligned[i]
        price_below_ema = close[i] < ema_50_1d_aligned[i]
        
        # Entry conditions
        long_setup = (close[i] > h4_1d_aligned[i]) and vol_confirm and price_above_ema
        short_setup = (close[i] < l4_1d_aligned[i]) and vol_confirm and price_below_ema
        
        # Exit conditions: mean reversion to opposite Camarilla level
        exit_long = close[i] < l4_1d_aligned[i]
        exit_short = close[i] > h4_1d_aligned[i]
        
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