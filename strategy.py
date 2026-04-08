#!/usr/bin/env python3
# [24998] daily_weekly_donchian_volume_ma_trend_v1
# Hypothesis: Daily Donchian(20) breakout with volume confirmation and weekly EMA trend filter.
# Long when price breaks above 20-day high with volume > 1.5x average and price > weekly EMA50.
# Short when price breaks below 20-day low with volume > 1.5x average and price < weekly EMA50.
# Exit when price returns to 10-day moving average.
# Weekly trend filter avoids counter-trend trades; volume confirmation ensures momentum.
# Designed for ~15-25 trades/year on daily timeframe to avoid fee drag while capturing strong trends.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "daily_weekly_donchian_volume_ma_trend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for EMA50 trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA50
    close_weekly = df_weekly['close'].values
    ema_50_weekly = np.full(len(close_weekly), np.nan)
    if len(close_weekly) >= 50:
        ema_50_weekly[49] = np.mean(close_weekly[:50])  # Simple average for first value
        alpha = 2.0 / (50 + 1)
        for i in range(50, len(close_weekly)):
            ema_50_weekly[i] = alpha * close_weekly[i] + (1 - alpha) * ema_50_weekly[i-1]
    
    # Calculate Donchian channels (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
    
    # Calculate 10-day moving average for exit
    ma_10 = np.full(n, np.nan)
    for i in range(10, n):
        ma_10[i] = np.mean(close[i-10:i])
    
    # Calculate volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    # Align weekly EMA50 to daily timeframe
    ema_50_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_50_weekly)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ma_10[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(ema_50_weekly_aligned[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        price = close[i]
        
        if position == 1:  # Long
            # Exit: price returns to 10-day MA
            if price <= ma_10[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: price returns to 10-day MA
            if price >= ma_10[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above Donchian high with volume expansion and above weekly EMA50
            if price > donchian_high[i] and vol_ratio > 1.5 and price > ema_50_weekly_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below Donchian low with volume expansion and below weekly EMA50
            elif price < donchian_low[i] and vol_ratio > 1.5 and price < ema_50_weekly_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals