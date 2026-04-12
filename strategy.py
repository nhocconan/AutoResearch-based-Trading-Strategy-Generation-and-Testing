#!/usr/bin/env python3
"""
12h_1d_Camarilla_Breakout_Volume_Regime_v1
Hypothesis: On 12h timeframe, buy when price breaks above daily Camarilla R3 level with volume > 2x average and close above 1d EMA(50) (uptrend); sell when price breaks below S3 with volume > 2x average and close below EMA(50) (downtrend). Exit when price returns to daily pivot point. Uses 1d timeframe for Camarilla levels and trend filter. Designed for low trade frequency by requiring multiple confirmations (level break, volume, trend). Works in bull/bear via 1d trend filter and mean-reversion exit at pivot.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_Camarilla_Breakout_Volume_Regime_v1"
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
    
    # Daily pivot point
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    r3_1d = close_1d + (range_1d * 1.1 / 4)
    s3_1d = close_1d - (range_1d * 1.1 / 4)
    r4_1d = close_1d + (range_1d * 1.1 / 2)
    s4_1d = close_1d - (range_1d * 1.1 / 2)
    
    # === 1D EMA(50) FOR TREND FILTER ===
    if len(close_1d) >= 50:
        ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    else:
        ema_50_1d = np.full_like(close_1d, np.nan)
    
    # Align data to 12h timeframe
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume average (12-period for 12h = ~6 days) for confirmation
    vol_avg = np.zeros(n)
    vol_sum = 0.0
    vol_count = 0
    for i in range(n):
        vol_sum += volume[i]
        vol_count += 1
        if i >= 12:
            vol_sum -= volume[i-12]
            vol_count -= 1
        if vol_count > 0:
            vol_avg[i] = vol_sum / vol_count
        else:
            vol_avg[i] = 0.0
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # start after warmup
        # Skip if indicators not available
        if (np.isnan(pivot_1d_aligned[i]) or np.isnan(r3_1d_aligned[i]) or 
            np.isnan(s3_1d_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or vol_avg[i] == 0.0):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: at least 2x average
        vol_confirm = volume[i] > 2.0 * vol_avg[i]
        
        # Trend filter: price above/below 1d EMA(50)
        price_above_ema = close[i] > ema_50_1d_aligned[i]
        price_below_ema = close[i] < ema_50_1d_aligned[i]
        
        # Entry conditions
        long_setup = (close[i] > r3_1d_aligned[i]) and vol_confirm and price_above_ema
        short_setup = (close[i] < s3_1d_aligned[i]) and vol_confirm and price_below_ema
        
        # Exit conditions: mean reversion to daily pivot
        exit_long = close[i] < pivot_1d_aligned[i]
        exit_short = close[i] > pivot_1d_aligned[i]
        
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