#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 12h data (HTF for direction) ===
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # 12h Donchian upper and lower bands (20 periods)
    high_20_12h = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    low_20_12h = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    donchian_upper_12h = align_htf_to_ltf(prices, df_12h, high_20_12h)
    donchian_lower_12h = align_htf_to_ltf(prices, df_12h, low_20_12h)
    
    # 12h EMA20 for trend filter
    close_12h_series = pd.Series(close_12h)
    ema_20_12h = close_12h_series.ewm(span=20, min_periods=20, adjust=False).mean().values
    ema_20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_20_12h)
    
    # === 4h data (for entry timing) ===
    df_4h = get_htf_data(prices, '4h')
    volume_4h = df_4h['volume'].values
    
    # Volume spike detection (4h)
    vol_ma_10_4h = pd.Series(volume_4h).rolling(window=10, min_periods=10).mean().values
    vol_ratio_4h = vol_ma_10_4h  # Will be used in loop with proper indexing
    
    # === Session filter: 08-20 UTC ===
    hours = prices.index.hour
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators have valid data
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper_12h[i]) or np.isnan(donchian_lower_12h[i]) or 
            np.isnan(ema_20_12h_aligned[i]) or np.isnan(vol_ratio_4h[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        price = close[i]
        upper_12h = donchian_upper_12h[i]
        lower_12h = donchian_lower_12h[i]
        ema_20_12h_val = ema_20_12h_aligned[i]
        vol_ratio_val = vol_ratio_4h[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when price closes below 12h Donchian lower
            if price < lower_12h:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when price closes above 12h Donchian upper
            if price > upper_12h:
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Only trade during session
            if in_session:
                # LONG: Price breaks above 12h Donchian upper AND above EMA20 (trend filter) AND volume spike
                if (price > upper_12h) and (price > ema_20_12h_val) and (vol_ratio_val > 2.0):
                    signals[i] = 0.25
                    position = 1
                    continue
                
                # SHORT: Price breaks below 12h Donchian lower AND below EMA20 (trend filter) AND volume spike
                elif (price < lower_12h) and (price < ema_20_12h_val) and (vol_ratio_val > 2.0):
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

name = "12h_Donchian_Breakout_EMA20_Volume_4h"
timeframe = "4h"
leverage = 1.0