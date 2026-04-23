#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d trend filter and volume confirmation.
Long when price breaks above upper Donchian channel and close > 1d EMA34 (uptrend) with volume > 1.5x average.
Short when price breaks below lower Donchian channel and close < 1d EMA34 (downtrend) with volume > 1.5x average.
Uses 12h timeframe to target 50-150 total trades over 4 years. Donchian channels provide price structure.
Volume confirmation ensures breakout conviction. Trend filter prevents counter-trend trades.
Works in both bull and bear markets by aligning with higher timeframe direction.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for Donchian channel calculation - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Donchian channels (20-period) from 1d data
    # Upper = max(high over last 20 periods)
    # Lower = min(low over last 20 periods)
    upper_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lower_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Load 1d data for EMA34 trend filter - ONCE before loop
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align HTF indicators to 12h timeframe
    upper_aligned = align_htf_to_ltf(prices, df_1d, upper_20)
    lower_aligned = align_htf_to_ltf(prices, df_1d, lower_20)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume average (20-period) on primary timeframe
    vol_ma_primary = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):  # Start after warmup period
        # Skip if data not ready
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(vol_ma_primary[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        upper_val = upper_aligned[i]
        lower_val = lower_aligned[i]
        ema34_val = ema34_1d_aligned[i]
        vol_ma_val = vol_ma_primary[i]
        
        # Get current price and volume
        price = close[i]
        vol_current = volume[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian AND price > 1d EMA34 (uptrend) AND volume confirmation
            if (price > upper_val and price > ema34_val and vol_current > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian AND price < 1d EMA34 (downtrend) AND volume confirmation
            elif (price < lower_val and price < ema34_val and vol_current > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price breaks below lower Donchian OR price breaks below 1d EMA34 (trend reversal)
                if price < lower_val or price < ema34_val:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price breaks above upper Donchian OR price breaks above 1d EMA34 (trend reversal)
                if price > upper_val or price > ema34_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_Donchian20_1dEMA34_Volume"
timeframe = "12h"
leverage = 1.0