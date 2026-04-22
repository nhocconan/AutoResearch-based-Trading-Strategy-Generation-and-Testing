#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h 12h Donchian breakout with volume confirmation and 12h EMA trend filter.
# Long when price breaks above 12h Donchian high + volume spike + price > 12h EMA50
# Short when price breaks below 12h Donchian low + volume spike + price < 12h EMA50
# Exit when price crosses back through 12h Donchian median or volume drops below 60% of average.
# Uses 12h timeframe for structure (trend and channels) and 4h for entry/exit timing.
# Target: 15-25 trades/year to minimize fee drag while capturing major moves.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 12h data for Donchian channels and EMA
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h Donchian channels (20-period)
    # Upper = max(high, lookback 20)
    # Lower = min(low, lookback 20)
    # Median = (upper + lower) / 2
    lookback = 20
    upper_12h = pd.Series(high_12h).rolling(window=lookback, min_periods=lookback).max().values
    lower_12h = pd.Series(low_12h).rolling(window=lookback, min_periods=lookback).min().values
    median_12h = (upper_12h + lower_12h) / 2
    
    # 12h EMA50 for trend filter
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align to 4h
    upper_aligned = align_htf_to_ltf(prices, df_12h, upper_12h)
    lower_aligned = align_htf_to_ltf(prices, df_12h, lower_12h)
    median_aligned = align_htf_to_ltf(prices, df_12h, median_12h)
    ema50_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume spike filter (20-period average)
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(upper_aligned[i]) or 
            np.isnan(lower_aligned[i]) or 
            np.isnan(median_aligned[i]) or 
            np.isnan(ema50_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        upper = upper_aligned[i]
        lower = lower_aligned[i]
        median = median_aligned[i]
        ema50 = ema50_aligned[i]
        
        # Volume filter: current volume > 2.0 * 20-day average (strict filter)
        vol_spike = vol > 2.0 * vol_ma
        
        if position == 0:
            # Long conditions: price breaks above 12h Donchian upper + volume spike + price > EMA50
            if price > upper and vol_spike and price > ema50:
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below 12h Donchian lower + volume spike + price < EMA50
            elif price < lower and vol_spike and price < ema50:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: price crosses back through 12h Donchian median or volume dries up
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price crosses below median or volume drops significantly
                if price < median or vol < 0.6 * vol_ma:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price crosses above median or volume drops significantly
                if price > median or vol < 0.6 * vol_ma:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_12h_Donchian20_Volume_EMA50_Trend"
timeframe = "4h"
leverage = 1.0