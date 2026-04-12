#!/usr/bin/env python3
"""
6h_1w_1d_WilliamsVixFix_v1
Hypothesis: On 6h timeframe, use weekly Williams Vix Fix (WVF) to identify extreme fear/greed
combined with daily trend filter. WVF > 0.8 indicates extreme fear (oversold) for long entries
in uptrend, WVF < 0.2 indicates extreme greed (overbought) for short entries in downtrend.
Exit when WVF returns to neutral zone (0.4-0.6). Works in bull/bear via daily trend filter
and mean-reversion exit. Low trade frequency expected due to extreme threshold requirements.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_1d_WilliamsVixFix_v1"
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
    
    # === WEEKLY WILLIAMS VIX FIX ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 22:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate WVF: (highest high - low) / (highest high - lowest low) * 100
    # Using 22-period lookback (approx 1 month)
    wvf = np.full(len(close_1w), np.nan)
    for i in range(21, len(close_1w)):
        highest_high = np.max(high_1w[i-21:i+1])
        lowest_low = np.min(low_1w[i-21:i+1])
        if highest_high > lowest_low:
            wvf[i] = (highest_high - low_1w[i]) / (highest_high - lowest_low) * 100
        else:
            wvf[i] = 50.0  # neutral when range is zero
    
    # === DAILY TREND FILTER ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Daily EMA(50) for trend
    ema_50 = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema_50[49] = np.mean(close_1d[:50])
        for i in range(50, len(close_1d)):
            ema_50[i] = (close_1d[i] * 0.0377) + (ema_50[i-1] * 0.9623)  # alpha = 2/(50+1)
    
    # Align data to 6h timeframe
    wvf_aligned = align_htf_to_ltf(prices, df_1w, wvf)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if indicators not available
        if (np.isnan(wvf_aligned[i]) or np.isnan(ema_50_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        wvf_val = wvf_aligned[i]
        ema_50_val = ema_50_aligned[i]
        close_val = close[i]
        
        # Entry conditions
        # Long: WVF > 80 (extreme fear) and price above daily EMA50 (uptrend)
        long_setup = (wvf_val > 80) and (close_val > ema_50_val)
        # Short: WVF < 20 (extreme greed) and price below daily EMA50 (downtrend)
        short_setup = (wvf_val < 20) and (close_val < ema_50_val)
        
        # Exit conditions: WVF returns to neutral zone (40-60)
        exit_long = wvf_val < 40
        exit_short = wvf_val > 60
        
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