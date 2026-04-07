#!/usr/bin/env python3
"""
1d Donchian Breakout with Weekly Trend Filter
Long when price breaks above 20-day Donchian high with weekly uptrend
Short when price breaks below 20-day Donchian low with weekly downtrend
Exit when price crosses 10-day EMA in opposite direction
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_donchian_breakout_1w_trend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === 20-day Donchian Channel ===
    # Upper: highest high of last 20 days
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    # Lower: lowest low of last 20 days
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 10-day EMA for exit ===
    ema_10 = pd.Series(close).ewm(span=10, min_periods=10, adjust=False).mean().values
    
    # === Weekly trend using EMA50 ===
    # Get weekly data ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    weekly_close = df_weekly['close'].values
    weekly_ema50 = pd.Series(weekly_close).ewm(span=50, min_periods=50, adjust=False).mean().values
    # Align to daily with proper shift
    weekly_ema50_aligned = align_htf_to_ltf(prices, df_weekly, weekly_ema50)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup period
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or \
           np.isnan(ema_10[i]) or np.isnan(weekly_ema50_aligned[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below 10-day EMA
            if close[i] < ema_10[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above 10-day EMA
            if close[i] > ema_10[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Weekly trend filter
            weekly_uptrend = weekly_ema50_aligned[i] > weekly_ema50_aligned[i-1]
            weekly_downtrend = weekly_ema50_aligned[i] < weekly_ema50_aligned[i-1]
            
            # Entry conditions
            if weekly_uptrend and close[i] > donchian_high[i]:
                # Break above Donchian high in weekly uptrend -> long
                position = 1
                signals[i] = 0.25
            elif weekly_downtrend and close[i] < donchian_low[i]:
                # Break below Donchian low in weekly downtrend -> short
                position = -1
                signals[i] = -0.25
    
    return signals