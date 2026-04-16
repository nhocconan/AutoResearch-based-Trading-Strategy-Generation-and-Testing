#!/usr/bin/env python3
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
    
    # === 12h data for trend filter ===
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # 12h EMA34 for trend filter
    ema_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # === 4h data for price channel ===
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Donchian(20) on 4h
    highest_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lowest_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    highest_4h_aligned = align_htf_to_ltf(prices, df_4h, highest_4h)
    lowest_4h_aligned = align_htf_to_ltf(prices, df_4h, lowest_4h)
    
    # === 1d data for volume spike detection ===
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    
    # Volume spike: current 4h volume > 2x 20-period 1d volume average (per 4h bar)
    # We need to map 1d volume average to 4h resolution
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    # For each 4h bar, compare its volume to the aligned 1d volume MA
    volume_spike = volume > (2.0 * vol_ma_1d_aligned)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators have valid data
    warmup = 60
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_12h_aligned[i]) or 
            np.isnan(highest_4h_aligned[i]) or 
            np.isnan(lowest_4h_aligned[i]) or 
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        ema_trend = ema_12h_aligned[i]
        upper_channel = highest_4h_aligned[i]
        lower_channel = lowest_4h_aligned[i]
        vol_spike = volume_spike[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when price returns to or below the lower Donchian band (mean reversion)
            if price <= lower_channel:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when price returns to or above the upper Donchian band (mean reversion)
            if price >= upper_channel:
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above upper Donchian(20) with volume spike and uptrend (12h EMA34)
            if price > upper_channel and vol_spike and price > ema_trend:
                signals[i] = 0.25
                position = 1
                continue
            
            # SHORT: Price breaks below lower Donchian(20) with volume spike and downtrend (12h EMA34)
            elif price < lower_channel and vol_spike and price < ema_trend:
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

name = "4h_Donchian20_VolumeSpike_EMA34Trend"
timeframe = "4h"
leverage = 1.0