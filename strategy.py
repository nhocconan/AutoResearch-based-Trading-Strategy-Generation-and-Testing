#!/usr/bin/env python3
# 12h_DonchianBreakout_Volume_Trend
# Hypothesis: 12h Donchian(20) breakout with volume confirmation and 1d EMA50 trend filter.
# Long when price breaks above upper band with volume > 1.5x avg and price > 1d EMA50.
# Short when price breaks below lower band with volume > 1.5x avg and price < 1d EMA50.
# Exit when price crosses below/above 12h EMA20 or reverses direction.
# Designed for low-frequency, high-conviction trades to avoid fee drag.

name = "12h_DonchianBreakout_Volume_Trend"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate 12h EMA20 for exit
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate Donchian channels (20-period)
    donchian_window = 20
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(donchian_window - 1, n):
        upper[i] = np.max(high[i - donchian_window + 1:i + 1])
        lower[i] = np.min(low[i - donchian_window + 1:i + 1])
    
    # Calculate average volume (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma[i] = np.mean(volume[i - 19:i + 1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(donchian_window - 1, 19, 50)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(ema20[i]) or 
            np.isnan(ema50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Volume confirmation: current volume > 1.5x average
            vol_confirm = volume[i] > 1.5 * vol_ma[i]
            
            # Long: price breaks above upper band + volume confirm + price > 1d EMA50
            if close[i] > upper[i] and vol_confirm and close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower band + volume confirm + price < 1d EMA50
            elif close[i] < lower[i] and vol_confirm and close[i] < ema50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price crosses below EMA20 or breaks below lower band
            if close[i] < ema20[i] or close[i] < lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price crosses above EMA20 or breaks above upper band
            if close[i] > ema20[i] or close[i] > upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals