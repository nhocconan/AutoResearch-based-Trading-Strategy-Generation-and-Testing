#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 12h data (HTF for trend and structure) ===
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # 12h Donchian(20) for structural breakout levels
    high_20_12h = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    low_20_12h = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    donchian_upper_12h = align_htf_to_ltf(prices, df_12h, high_20_12h)
    donchian_lower_12h = align_htf_to_ltf(prices, df_12h, low_20_12h)
    
    # 12h EMA50 for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # === 1d data (HTF for volatility and volume context) ===
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    
    # 1d volume moving average for spike detection
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1d = volume_1d / vol_ma_20_1d
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators have valid data
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper_12h[i]) or np.isnan(donchian_lower_12h[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ratio_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        upper_12h = donchian_upper_12h[i]
        lower_12h = donchian_lower_12h[i]
        ema_50_12h_val = ema_50_12h_aligned[i]
        vol_ratio_val = vol_ratio_1d_aligned[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when price closes below 12h Donchian lower OR trend turns bearish
            if (price < lower_12h) or (price < ema_50_12h_val):
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when price closes above 12h Donchian upper OR trend turns bullish
            if (price > upper_12h) or (price > ema_50_12h_val):
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above 12h Donchian upper AND above EMA50 (uptrend) 
            # AND volume spike (institutional interest)
            if (price > upper_12h) and (price > ema_50_12h_val) and (vol_ratio_val > 1.8):
                signals[i] = 0.25
                position = 1
                continue
            
            # SHORT: Price breaks below 12h Donchian lower AND below EMA50 (downtrend) 
            # AND volume spike (institutional interest)
            elif (price < lower_12h) and (price < ema_50_12h_val) and (vol_ratio_val > 1.8):
                signals[i] = -0.25
                position = -1
                continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_Donchian_Breakout_EMA50_12h_Volume"
timeframe = "6h"
leverage = 1.0