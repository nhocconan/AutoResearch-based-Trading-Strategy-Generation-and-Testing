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
    
    # === 6h data (primary timeframe) ===
    df_6h = get_htf_data(prices, '6h')
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    volume_6h = df_6h['volume'].values
    
    # Calculate ATR on 6h
    tr_6h = np.maximum(high_6h - low_6h,
                       np.maximum(np.abs(high_6h - np.roll(close_6h, 1)),
                                  np.abs(low_6h - np.roll(close_6h, 1))))
    tr_6h[0] = high_6h[0] - low_6h[0]
    atr_6h = pd.Series(tr_6h).rolling(window=14, min_periods=14).mean().values
    atr_6h_aligned = align_htf_to_ltf(prices, df_6h, atr_6h)
    
    # === 12h data (HTF) ===
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # === 12h EMA34 for trend filter ===
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # === 12h Donchian Channel (20) for breakout signals ===
    highest_20_12h = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    lowest_20_12h = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    donchian_upper_12h = highest_20_12h
    donchian_lower_12h = lowest_20_12h
    donchian_upper_12h_aligned = align_htf_to_ltf(prices, df_12h, donchian_upper_12h)
    donchian_lower_12h_aligned = align_htf_to_ltf(prices, df_12h, donchian_lower_12h)
    
    # === 6h Volume spike detection ===
    vol_ma_20 = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume_6h / vol_ma_20
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators have valid data
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_6h_aligned[i]) or np.isnan(ema_34_12h_aligned[i]) or 
            np.isnan(donchian_upper_12h_aligned[i]) or np.isnan(donchian_lower_12h_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close_6h[i]
        atr_6h_val = atr_6h_aligned[i]
        ema_34_12h_val = ema_34_12h_aligned[i]
        donchian_upper_12h_val = donchian_upper_12h_aligned[i]
        donchian_lower_12h_val = donchian_lower_12h_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when price closes below 12h EMA34 OR volatility regime shifts (via ATR expansion)
            if (price < ema_34_12h_val) or (atr_6h_val > 1.5 * atr_6h):  # Using current ATR vs smoothed
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when price closes above 12h EMA34 OR volatility regime shifts
            if (price > ema_34_12h_val) or (atr_6h_val > 1.5 * atr_6h):
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above 12h Donchian upper AND above 12h EMA34 AND volume spike
            if (price > donchian_upper_12h_val) and (price > ema_34_12h_val) and (vol_ratio_val > 2.0):
                signals[i] = 0.25
                position = 1
                continue
            
            # SHORT: Price breaks below 12h Donchian lower AND below 12h EMA34 AND volume spike
            elif (price < donchian_lower_12h_val) and (price < ema_34_12h_val) and (vol_ratio_val > 2.0):
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

name = "6h_12h_Donchian_EMA34_VolumeBreakout"
timeframe = "6h"
leverage = 1.0