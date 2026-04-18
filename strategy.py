#!/usr/bin/env python3
"""
Hypothesis: 1h strategy using 4h Donchian breakout (20) + 1d EMA(50) trend filter + volume confirmation. 
4h Donchian provides clear trend structure, 1d EMA filters for higher timeframe direction, 
volume ensures breakout validity. Designed for 20-40 trades/year on 1h to minimize fee drag.
Works in bull markets (buy upper band breakouts in uptrend) and bear markets (sell lower band breakouts in downtrend).
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
    
    # Get 4h data for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate Donchian channels (20-period) on 4h
    donch_high_4h = np.full(len(high_4h), np.nan)
    donch_low_4h = np.full(len(low_4h), np.nan)
    
    for i in range(19, len(high_4h)):
        donch_high_4h[i] = np.max(high_4h[i-19:i+1])
        donch_low_4h[i] = np.min(low_4h[i-19:i+1])
    
    # Align Donchian channels to 1h timeframe
    donch_high_1h = align_htf_to_ltf(prices, df_4h, donch_high_4h)
    donch_low_1h = align_htf_to_ltf(prices, df_4h, donch_low_4h)
    
    # Get 1d data for EMA(50) trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA(50) on 1d close
    ema_50_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema_50_1d[49] = np.mean(close_1d[:50])
        for i in range(50, len(close_1d)):
            ema_50_1d[i] = (close_1d[i] * 2/51) + (ema_50_1d[i-1] * 49/51)
    
    # Align 1d EMA to 1h timeframe
    ema_50_1h = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    # Session filter: 08-20 UTC (inclusive)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(40, 20)  # need Donchian, EMA, volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donch_high_1h[i]) or np.isnan(donch_low_1h[i]) or 
            np.isnan(ema_50_1h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        vol_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        # Trend filter: price above/below 1d EMA50
        trend_up = close[i] > ema_50_1h[i]
        trend_down = close[i] < ema_50_1h[i]
        
        if position == 0:
            # Long entry: close above 4h Donchian upper band with volume and uptrend
            if (close[i] > donch_high_1h[i] and 
                vol_confirmed and 
                trend_up):
                signals[i] = 0.20
                position = 1
            # Short entry: close below 4h Donchian lower band with volume and downtrend
            elif (close[i] < donch_low_1h[i] and 
                  vol_confirmed and 
                  trend_down):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: close below 4h Donchian lower band or reverse signal
            if close[i] < donch_low_1h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: close above 4h Donchian upper band or reverse signal
            if close[i] > donch_high_1h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Donchian20_1dEMA50_Volume"
timeframe = "1h"
leverage = 1.0