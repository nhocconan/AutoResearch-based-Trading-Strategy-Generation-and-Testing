#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and 1d trend filter.
# Long when price breaks above Donchian high, volume > 1.5x 20-period average, and price > 1d EMA50.
# Short when price breaks below Donchian low, volume > 1.5x 20-period average, and price < 1d EMA50.
# Uses discrete position size (0.25) to minimize churn. Designed for 4h timeframe
# to capture breakouts with trend and volume confirmation. Target: 25-50 trades/year per symbol (~100-200 total over 4 years).
name = "4h_Donchian20_Volume_Trend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA50 on daily
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 4h
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Donchian channel (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)  # Ensure Donchian and EMA50 are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        donch_high = donchian_high[i]
        donch_low = donchian_low[i]
        ema_50_val = ema_50_aligned[i]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        
        # Volume confirmation threshold
        volume_confirmed = vol > 1.5 * vol_ma
        
        if position == 0:
            # Enter long if price breaks above Donchian high, volume confirmed, and above EMA50
            if price > donch_high and volume_confirmed and price > ema_50_val:
                signals[i] = 0.25
                position = 1
            # Enter short if price breaks below Donchian low, volume confirmed, and below EMA50
            elif price < donch_low and volume_confirmed and price < ema_50_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long when price crosses below Donchian low or trend reverses
            if price < donch_low or price < ema_50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when price crosses above Donchian high or trend reverses
            if price > donch_high or price > ema_50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals