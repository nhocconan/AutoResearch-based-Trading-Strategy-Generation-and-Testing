#!/usr/bin/env python3
# 12h_1d_donchian_volume_breakout_v1
# Hypothesis: 12-hour Donchian breakout with volume confirmation and 1-day trend filter.
# Long when price breaks above 20-period Donchian high + volume > 1.5x 20-period average + price > 1-day EMA50.
# Short when price breaks below 20-period Donchian low + volume > 1.5x 20-period average + price < 1-day EMA50.
# Exit when price crosses back below/above 10-period EMA or volume drops below average.
# Uses 1-day EMA for trend filter and 12-hour Donchian/volume for entry timing.
# Target: 20-40 trades/year to minimize fee dust while capturing strong breakouts.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_donchian_volume_breakout_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 12-period EMA for exit
    ema_12 = np.zeros(n)
    ema_12[:] = np.nan
    ema_12[11] = np.mean(close[:12])
    for i in range(12, n):
        ema_12[i] = close[i] * 0.1538 + ema_12[i-1] * 0.8462
    
    # Calculate 20-period Donchian channels
    donchian_high = np.zeros(n)
    donchian_low = np.zeros(n)
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
    
    # Calculate 20-period average volume
    avg_volume = np.zeros(n)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    # Get 1-day EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_1d_50 = np.zeros(len(close_1d))
    ema_1d_50[:] = np.nan
    ema_1d_50[49] = np.mean(close_1d[:50])
    for i in range(50, len(close_1d)):
        ema_1d_50[i] = close_1d[i] * 0.0377 + ema_1d_50[i-1] * 0.9623
    
    ema_1d_50_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_50)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        price = close[i]
        vol = volume[i]
        donch_high = donchian_high[i]
        donch_low = donchian_low[i]
        avg_vol = avg_volume[i]
        ema12 = ema_12[i]
        ema1d = ema_1d_50_aligned[i]
        
        if np.isnan(donch_high) or np.isnan(donch_low) or np.isnan(avg_vol) or np.isnan(ema12) or np.isnan(ema1d):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        vol_surge = vol > 1.5 * avg_vol
        
        if position == 1:  # Long
            # Exit: price below 12 EMA OR volume drops below average
            if price < ema12 or vol < avg_vol:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: price above 12 EMA OR volume drops below average
            if price > ema12 or vol < avg_vol:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry: Donchian breakout + volume surge + 1-day EMA filter
            if price > donch_high and vol_surge and price > ema1d:
                position = 1
                signals[i] = 0.25
            elif price < donch_low and vol_surge and price < ema1d:
                position = -1
                signals[i] = -0.25
    
    return signals