#!/usr/bin/env python3
"""
12h Donchian Breakout with Weekly EMA Trend Filter and Volume Confirmation
Long: Price breaks above Donchian(20) high + price > weekly EMA50 + volume > 1.5x avg
Short: Price breaks below Donchian(20) low + price < weekly EMA50 + volume > 1.5x avg
Exit: Opposite Donchian breakout
Uses 12h for entry timing, 1w EMA for trend filter, volume for confirmation.
Designed to capture trends in both bull and bear markets with low frequency.
Target: 20-40 total trades over 4 years (5-10/year)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need enough data for indicators
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 12h Donchian Channel (20-period)
    period20_high = pd.Series(high).rolling(window=20, min_periods=20).max()
    period20_low = pd.Series(low).rolling(window=20, min_periods=20).min()
    donchian_high = period20_high.values
    donchian_low = period20_low.values
    
    # 12h Average Volume (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get 1w data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA50
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w EMA50 to 12h
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 50  # Need EMA50 and Donchian calculations
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(avg_volume[i]) or np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = avg_volume[i]
        donch_high = donchian_high[i]
        donch_low = donchian_low[i]
        ema_trend = ema_50_1w_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 20-bar average
        vol_confirm = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long: Break above Donchian high + above weekly EMA + volume confirmation
            if price > donch_high and price > ema_trend and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Break below Donchian low + below weekly EMA + volume confirmation
            elif price < donch_low and price < ema_trend and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price breaks below Donchian low
            if price < donch_low:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price breaks above Donchian high
            if price > donch_high:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian_WeeklyEMA_Volume"
timeframe = "12h"
leverage = 1.0