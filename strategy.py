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
    
    # === 12h Donchian Channel (20-period) - Primary Signal ===
    df_12h = get_htf_data(prices, '12h')
    donchian_high = pd.Series(df_12h['high']).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(df_12h['low']).rolling(window=20, min_periods=20).min().values
    
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    
    # === Weekly High/Low for Trend Context (1w HTF) ===
    df_1w = get_htf_data(prices, '1w')
    weekly_high = pd.Series(df_1w['high']).rolling(window=4, min_periods=4).max().values  # ~1 month
    weekly_low = pd.Series(df_1w['low']).rolling(window=4, min_periods=4).min().values
    
    weekly_high_aligned = align_htf_to_ltf(prices, df_1w, weekly_high)
    weekly_low_aligned = align_htf_to_ltf(prices, df_1w, weekly_low)
    
    # === Volume Confirmation (12h volume MA) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)  # Strong volume spike to reduce trades
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators have valid data
    warmup = 100  # Need sufficient data for all indicators
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(weekly_high_aligned[i]) or np.isnan(weekly_low_aligned[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        vol_spike = volume_spike[i]
        
        # === EXIT LOGIC: Exit when price crosses midline of Donchian ===
        midline = (donchian_high_aligned[i] + donchian_low_aligned[i]) / 2
        
        if position == 1:  # Long position
            # Exit when price crosses back below midline
            if price < midline:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when price crosses back above midline
            if price > midline:
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Determine trend bias from weekly levels
            weekly_mid = (weekly_high_aligned[i] + weekly_low_aligned[i]) / 2
            bullish_bias = price > weekly_mid
            bearish_bias = price < weekly_mid
            
            # LONG: Price breaks above Donchian high with volume confirmation and bullish weekly bias
            if price > donchian_high_aligned[i] and vol_spike and bullish_bias:
                signals[i] = 0.25
                position = 1
                continue
            
            # SHORT: Price breaks below Donchian low with volume confirmation and bearish weekly bias
            elif price < donchian_low_aligned[i] and vol_spike and bearish_bias:
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

name = "12h_Donchian20_1w_Trend_Volume"
timeframe = "12h"
leverage = 1.0