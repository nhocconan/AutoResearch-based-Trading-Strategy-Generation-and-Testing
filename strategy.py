#!/usr/bin/env python3
# [24986] 4h_1d_donchian_volume_trend_v1
# Hypothesis: 4-hour Donchian(20) breakout with 1-day trend filter (price > EMA50) and volume confirmation.
# Long when price breaks above 20-period high with volume > 2.0x average and price > 1-day EMA50.
# Short when price breaks below 20-period low with volume > 2.0x average and price < 1-day EMA50.
# Exit when price returns to 10-period moving average.
# Uses 1-day EMA50 for trend bias, effective in both trending and ranging markets.
# Designed to generate ~20-50 trades/year to avoid fee drag while capturing strong trends.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_donchian_volume_trend_v1"
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
    
    # Get 1-day data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate 1-day EMA50
    close_1d = df_1d['close'].values
    ema50_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema = np.zeros(len(close_1d))
        ema[0] = close_1d[0]
        alpha = 2.0 / (50 + 1)
        for i in range(1, len(close_1d)):
            ema[i] = alpha * close_1d[i] + (1 - alpha) * ema[i-1]
        ema50_1d[49:] = ema[49:]
    
    # Calculate Donchian channels (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
    
    # Calculate 10-period moving average for exit
    ma_10 = np.full(n, np.nan)
    for i in range(10, n):
        ma_10[i] = np.mean(close[i-10:i])
    
    # Calculate volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    # Align 1-day EMA50 to 4-hour timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ma_10[i]) or np.isnan(vol_ma[i]) or np.isnan(ema50_1d_aligned[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        price = close[i]
        
        if position == 1:  # Long
            # Exit: price returns to 10-period MA
            if price <= ma_10[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: price returns to 10-period MA
            if price >= ma_10[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above Donchian high with volume expansion and above EMA50
            if price > donchian_high[i] and vol_ratio > 2.0 and price > ema50_1d_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below Donchian low with volume expansion and below EMA50
            elif price < donchian_low[i] and vol_ratio > 2.0 and price < ema50_1d_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals