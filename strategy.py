#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d trend filter and volume confirmation.
Long when price breaks above Donchian upper band (20-period high) and close > 1d EMA50 (uptrend) with volume > 1.5x average.
Short when price breaks below Donchian lower band (20-period low) and close < 1d EMA50 (downtrend) with volume > 1.5x average.
Uses 4h timeframe to target 75-200 total trades over 4 years. Donchian provides clear price channel structure.
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
    
    # Load 1d data for EMA50 trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to 4h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate Donchian channels (20-period) on primary timeframe
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume average (20-period) on primary timeframe
    vol_ma_primary = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):  # Start after warmup period
        # Skip if data not ready
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(high_roll[i]) or 
            np.isnan(low_roll[i]) or np.isnan(vol_ma_primary[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        upper_band = high_roll[i]
        lower_band = low_roll[i]
        ema50_val = ema50_1d_aligned[i]
        vol_ma_val = vol_ma_primary[i]
        
        # Get current price and volume
        price = close[i]
        vol_current = volume[i]
        
        if position == 0:
            # Long: price breaks above Donchian upper band AND price > 1d EMA50 (uptrend) AND volume confirmation
            if (price > upper_band and price > ema50_val and vol_current > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower band AND price < 1d EMA50 (downtrend) AND volume confirmation
            elif (price < lower_band and price < ema50_val and vol_current > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price breaks below Donchian lower band OR price breaks below 1d EMA50 (trend reversal)
                if price < lower_band or price < ema50_val:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price breaks above Donchian upper band OR price breaks above 1d EMA50 (trend reversal)
                if price > upper_band or price > ema50_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Donchian20_1dEMA50_Volume"
timeframe = "4h"
leverage = 1.0