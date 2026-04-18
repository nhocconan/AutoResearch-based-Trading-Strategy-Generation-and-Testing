#!/usr/bin/env python3
"""
1d Donchian(20) breakout with weekly trend filter (1w EMA(50)) and volume confirmation.
Donchian channels identify breakout points with clear structure. Weekly EMA filters
trend direction to trade with the higher timeframe momentum. Volume confirmation
ensures breakouts have conviction. Designed for 10-20 trades/year to minimize fee drag.
Works in bull markets (buy upper band breaks in uptrend) and bear markets
(sell lower band breaks in downtrend).
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian channel calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Donchian channels (20-period) on 1d data
    donchian_high = np.full(len(high_1d), np.nan)
    donchian_low = np.full(len(low_1d), np.nan)
    
    for i in range(20, len(high_1d)):
        donchian_high[i] = np.max(high_1d[i-20:i])
        donchian_low[i] = np.min(low_1d[i-20:i])
    
    # Align Donchian levels to 1d timeframe (already aligned via index)
    donchian_high_1d = donchian_high
    donchian_low_1d = donchian_low
    
    # Get weekly data for EMA(50) trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate EMA(50) on weekly close
    ema_50_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 50:
        ema_50_1w[49] = np.mean(close_1w[:50])
        for i in range(50, len(close_1w)):
            ema_50_1w[i] = (close_1w[i] * 2/51) + (ema_50_1w[i-1] * 49/51)
    
    # Align weekly EMA to 1d timeframe
    ema_50_1w_1d = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)  # need Donchian, EMA, volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_high_1d[i]) or np.isnan(donchian_low_1d[i]) or 
            np.isnan(ema_50_1w_1d[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        vol_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        # Trend filter: price above/below weekly EMA50
        trend_up = close[i] > ema_50_1w_1d[i]
        trend_down = close[i] < ema_50_1w_1d[i]
        
        if position == 0:
            # Long entry: close above Donchian upper band with volume and uptrend
            if (close[i] > donchian_high_1d[i] and 
                vol_confirmed and 
                trend_up):
                signals[i] = 0.25
                position = 1
            # Short entry: close below Donchian lower band with volume and downtrend
            elif (close[i] < donchian_low_1d[i] and 
                  vol_confirmed and 
                  trend_down):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: close below Donchian lower band or reverse signal
            if close[i] < donchian_low_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: close above Donchian upper band or reverse signal
            if close[i] > donchian_high_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_1wEMA50_Volume"
timeframe = "1d"
leverage = 1.0