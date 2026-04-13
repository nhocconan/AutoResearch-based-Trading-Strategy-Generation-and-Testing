#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Donchian(20) breakout with 12-hour EMA50 trend filter and volume confirmation.
# Long: price breaks above Donchian Upper + price above 12h EMA50 + volume > 1.8x avg volume
# Short: price breaks below Donchian Lower + price below 12h EMA50 + volume > 1.8x avg volume
# Donchian levels calculated from 4h data: Upper = max(high, 20), Lower = min(low, 20)
# Trend filter: only take longs when price > EMA50, shorts when price < EMA50
# Volume confirmation reduces false breakouts
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe
# Works in both bull and bear markets by using 12h EMA as trend filter

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4-hour data for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Donchian channels (20-period)
    donch_upper_4h = np.full(len(high_4h), np.nan)
    donch_lower_4h = np.full(len(low_4h), np.nan)
    for i in range(20, len(high_4h)):
        donch_upper_4h[i] = np.max(high_4h[i-20:i])
        donch_lower_4h[i] = np.min(low_4h[i-20:i])
    
    # Align 4h Donchian to 4h timeframe (no alignment needed as we're using 4h as primary)
    donch_upper = donch_upper_4h
    donch_lower = donch_lower_4h
    
    # 12-hour EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 5:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h EMA to 4h timeframe
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Average volume (20-period = 20 periods) for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if any required data is not ready
        if (np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        upper = donch_upper[i]
        lower = donch_lower[i]
        ema_trend = ema_50_12h_aligned[i]
        
        # Volume confirmation: current volume > 1.8x average volume
        volume_confirm = vol > 1.8 * avg_vol
        
        if position == 0:
            # Long: break above Donchian Upper + above EMA50 + volume confirmation
            if (price > upper and 
                price > ema_trend and
                volume_confirm):
                position = 1
                signals[i] = position_size
            # Short: break below Donchian Lower + below EMA50 + volume confirmation
            elif (price < lower and 
                  price < ema_trend and
                  volume_confirm):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below Donchian Lower or below EMA50
            if (price < lower or
                price < ema_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above Donchian Upper or above EMA50
            if (price > upper or
                price > ema_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_12h_Donchian_EMA_Volume"
timeframe = "4h"
leverage = 1.0