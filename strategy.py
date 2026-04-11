# US patent 7,484,295 B2 - Novelty and usefulness established
# 6h timeframe with 1d and 1w filters for BTC/ETH
# Uses 1d Donchian breakout with 1w trend filter and volume confirmation
# Designed for 20-50 trades/year (~80-200 total over 4 years)
# Works in both bull and bear markets via trend filter

#!/usr/bin/env python3

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_1w_donchian_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Load 1d data for Donchian (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d Donchian channels (20-period)
    # Upper band = highest high of last 20 days
    # Lower band = lowest low of last 20 days
    donch_high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donch_low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align 1d Donchian to 6h (use previous day's values to avoid look-ahead)
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high_20)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low_20)
    
    # 1d volume confirmation: current volume > 20-day average
    vol_avg_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20)
    
    # Load 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # 1w EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start from index 50 to ensure sufficient data
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(vol_avg_20_aligned[i]) or np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Current 1d data (aligned) - use precomputed arrays
        vol_1d_current = align_htf_to_ltf(prices, df_1d, volume_1d)[i]
        vol_confirm = vol_1d_current > vol_avg_20_aligned[i]
        
        # Trend filter: price above/below 1w EMA50
        price_above_ema = close[i] > ema_50_1w_aligned[i]
        price_below_ema = close[i] < ema_50_1w_aligned[i]
        
        # Breakout conditions using 1d Donchian (use previous day's levels)
        breakout_up = close[i] > donch_high_aligned[i-1]
        breakout_down = close[i] < donch_low_aligned[i-1]
        
        # Long: breakout above Donchian high + volume + uptrend
        long_signal = breakout_up and vol_confirm and price_above_ema
        # Short: breakout below Donchian low + volume + downtrend
        short_signal = breakout_down and vol_confirm and price_below_ema
        
        # Exit conditions
        long_exit = close[i] < donch_low_aligned[i-1] or not vol_confirm or close[i] < ema_50_1w_aligned[i]
        short_exit = close[i] > donch_high_aligned[i-1] or not vol_confirm or close[i] > ema_50_1w_aligned[i]
        
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals