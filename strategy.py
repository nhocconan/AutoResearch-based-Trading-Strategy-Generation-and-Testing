#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot long/short with 4h trend filter and volume confirmation.
# Long: price > H3 (Camarilla) + price > 4h EMA20 + volume > 1.5x avg volume
# Short: price < L3 (Camarilla) + price < 4h EMA20 + volume > 1.5x avg volume
# Uses Camarilla pivots from prior day for intraday structure.
# 4h EMA20 filters trades to align with higher timeframe trend.
# Volume confirmation reduces false breakouts.
# Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe.
# Works in both bull and bear markets by using 4h EMA as trend filter.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    camarilla_h3 = np.full(len(close_1d), np.nan)
    camarilla_l3 = np.full(len(close_1d), np.nan)
    
    for i in range(1, len(close_1d)):
        # Previous day's data
        prev_close = close_1d[i-1]
        prev_high = high_1d[i-1]
        prev_low = low_1d[i-1]
        
        # Camarilla formulas
        rang = prev_high - prev_low
        camarilla_h3[i] = prev_close + rang * 1.1 / 6
        camarilla_l3[i] = prev_close - rang * 1.1 / 6
    
    # Align Camarilla levels to 1h timeframe
    camarilla_h3_1h = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_1h = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # 4h EMA20 for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_1h = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # Average volume (20-period) for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.20  # 20% position size
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    for i in range(20, n):
        # Skip if any required data is not ready
        if (np.isnan(camarilla_h3_1h[i]) or np.isnan(camarilla_l3_1h[i]) or 
            np.isnan(ema_20_4h_1h[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Apply session filter (08-20 UTC)
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        h3 = camarilla_h3_1h[i]
        l3 = camarilla_l3_1h[i]
        ema_trend = ema_20_4h_1h[i]
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirm = vol > 1.5 * avg_vol
        
        if position == 0:
            # Long: price > H3 + price > 4h EMA20 + volume confirmation
            if (price > h3 and 
                price > ema_trend and
                volume_confirm):
                position = 1
                signals[i] = position_size
            # Short: price < L3 + price < 4h EMA20 + volume confirmation
            elif (price < l3 and 
                  price < ema_trend and
                  volume_confirm):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below L3 or below 4h EMA20
            if (price < l3 or
                price < ema_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above H3 or above 4h EMA20
            if (price > h3 or
                price > ema_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1h_4h_1d_Camarilla_EMA_Volume"
timeframe = "1h"
leverage = 1.0