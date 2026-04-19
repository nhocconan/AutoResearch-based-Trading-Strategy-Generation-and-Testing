#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and 1d EMA(50) trend filter.
# Long when: price breaks above Donchian high(20), volume > 1.5x 20-period average, and price > 1d EMA(50)
# Short when: price breaks below Donchian low(20), volume > 1.5x 20-period average, and price < 1d EMA(50)
# Exit when price crosses the Donchian midpoint or trend reverses.
# Designed for ~20-30 trades/year per symbol.
name = "4h_Donchian20_Volume_EMA50_Trend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_1d_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_50_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_50)
    
    # Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    mid_20 = (high_20 + low_20) / 2.0
    
    # Volume confirmation: 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or np.isnan(ema_1d_50_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        donch_high = high_20[i]
        donch_low = low_20[i]
        donch_mid = mid_20[i]
        vol_ma = vol_ma_20[i]
        ema_1d = ema_1d_50_aligned[i]
        
        if position == 0:
            # Long: break above Donchian high, high volume, price above 1d EMA50
            if price > donch_high and vol > 1.5 * vol_ma and price > ema_1d:
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low, high volume, price below 1d EMA50
            elif price < donch_low and vol > 1.5 * vol_ma and price < ema_1d:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below Donchian midpoint or trend reverses (price < 1d EMA50)
            if price < donch_mid or price < ema_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above Donchian midpoint or trend reverses (price > 1d EMA50)
            if price > donch_mid or price > ema_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals