#!/usr/bin/env python3
"""
4h_1d_Donchian_Breakout_Trend_v1
Hypothesis: On 4h timeframe, trade breakouts of daily Donchian channels (20-period) with 1d trend filter.
Long when price breaks above 20-day high with 1d uptrend; short when breaks below 20-day low with 1d downtrend.
Exit when price returns to the 20-day midpoint. Uses volume confirmation to avoid false breakouts.
Designed for low trade frequency (20-40/year) by requiring multi-timeframe confluence.
Works in bull/bear via 1d trend filter and mean-reversion exit at channel midpoint.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_Donchian_Breakout_Trend_v1"
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
    
    # === DAILY DONCHIAN CHANNEL (20-period) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 20-day high and low
    high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Midpoint for exit
    mid_20 = (high_20 + low_20) / 2
    
    # === 1D EMA(50) FOR TREND FILTER ===
    if len(close_1d) >= 50:
        ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    else:
        ema_50_1d = np.full_like(close_1d, np.nan)
    
    # Align data to 4h timeframe
    high_20_aligned = align_htf_to_ltf(prices, df_1d, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_1d, low_20)
    mid_20_aligned = align_htf_to_ltf(prices, df_1d, mid_20)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume average (20-period for 4h = ~10 days) for confirmation
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
        if (np.isnan(high_20_aligned[i]) or np.isnan(low_20_aligned[i]) or 
            np.isnan(mid_20_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or vol_avg[i] == 0.0):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: at least 1.5x average
        vol_confirm = volume[i] > 1.5 * vol_avg[i]
        
        # Trend filter: price above/below 1d EMA(50)
        price_above_ema = close[i] > ema_50_1d_aligned[i]
        price_below_ema = close[i] < ema_50_1d_aligned[i]
        
        # Entry conditions
        long_setup = (close[i] > high_20_aligned[i]) and vol_confirm and price_above_ema
        short_setup = (close[i] < low_20_aligned[i]) and vol_confirm and price_below_ema
        
        # Exit conditions: mean reversion to 20-day midpoint
        exit_long = close[i] < mid_20_aligned[i]
        exit_short = close[i] > mid_20_aligned[i]
        
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