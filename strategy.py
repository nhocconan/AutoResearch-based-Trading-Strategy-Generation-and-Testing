#!/usr/bin/env python3
"""
1d Donchian Breakout with Weekly Trend and Volume Confirmation
Long: Close breaks above Donchian(20) high AND price above 20-week EMA AND volume > 1.5x 20-day volume average
Short: Close breaks below Donchian(20) low AND price below 20-week EMA AND volume > 1.5x 20-day volume average
Exit: Close crosses back below Donchian(20) midpoint (for long) or above midpoint (for short)
Target: 8-20 trades/year per symbol (32-80 total over 4 years)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 20-week EMA for trend direction
    ema_20_1w = pd.Series(df_1w['close'].values).ewm(span=20, adjust=False).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Calculate Donchian channels (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (high_roll + low_roll) / 2.0
    
    # Calculate 20-day volume average for volume filter
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 20  # Donchian needs 20 periods
    
    for i in range(start_idx, n):
        if (np.isnan(ema_20_1w_aligned[i]) or np.isnan(high_roll[i]) or 
            np.isnan(low_roll[i]) or np.isnan(vol_avg[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_avg_val = vol_avg[i]
        ema_1w_val = ema_20_1w_aligned[i]
        donchian_high = high_roll[i]
        donchian_low = low_roll[i]
        donchian_mid_val = donchian_mid[i]
        
        if position == 0:
            # Long: Close breaks above Donchian high + price above 20-week EMA + volume > 1.5x avg
            if price > donchian_high and price > ema_1w_val and vol > 1.5 * vol_avg_val:
                signals[i] = 0.25
                position = 1
            # Short: Close breaks below Donchian low + price below 20-week EMA + volume > 1.5x avg
            elif price < donchian_low and price < ema_1w_val and vol > 1.5 * vol_avg_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Close crosses below Donchian midpoint
            if price < donchian_mid_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Close crosses above Donchian midpoint
            if price > donchian_mid_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian_WeeklyTrend_Volume"
timeframe = "1d"
leverage = 1.0