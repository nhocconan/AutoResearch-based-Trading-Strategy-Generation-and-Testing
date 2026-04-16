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
    
    # === 4h data (primary timeframe) ===
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Calculate ATR on 4h
    tr_4h = np.maximum(high_4h - low_4h,
                       np.maximum(np.abs(high_4h - np.roll(close_4h, 1)),
                                  np.abs(low_4h - np.roll(close_4h, 1))))
    tr_4h[0] = high_4h[0] - low_4h[0]
    atr_4h = pd.Series(tr_4h).rolling(window=14, min_periods=14).mean().values
    atr_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_4h)
    
    # === 1d data (HTF) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate ATR on 1d
    tr_1d = np.maximum(high_1d - low_1d,
                       np.maximum(np.abs(high_1d - np.roll(close_1d, 1)),
                                  np.abs(low_1d - np.roll(close_1d, 1))))
    tr_1d[0] = high_1d[0] - low_1d[0]
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # === 1d Bollinger Bands (20, 2) ===
    sma_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_band = sma_20 + (2 * std_20)
    lower_band = sma_20 - (2 * std_20)
    upper_band_aligned = align_htf_to_ltf(prices, df_1d, upper_band)
    lower_band_aligned = align_htf_to_ltf(prices, df_1d, lower_band)
    
    # === 1d Volume SMA (20) ===
    volume_sma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators have valid data
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_4h_aligned[i]) or np.isnan(atr_1d_aligned[i]) or 
            np.isnan(upper_band_aligned[i]) or np.isnan(lower_band_aligned[i]) or
            np.isnan(volume_sma_20_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close_4h[i]
        atr_4h_val = atr_4h_aligned[i]
        upper_band_val = upper_band_aligned[i]
        lower_band_val = lower_band_aligned[i]
        volume_val = volume_4h[i]
        volume_sma_val = volume_sma_20_aligned[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when price closes below lower Bollinger Band OR trailing stop hit
            if (price < lower_band_val) or (price < (highest_since_entry - 2.0 * atr_4h_val)):
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when price closes above upper Bollinger Band OR trailing stop hit
            if (price > upper_band_val) or (price > (lowest_since_entry + 2.0 * atr_4h_val)):
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Track highest/lowest since entry for trailing stop
            highest_since_entry = price
            lowest_since_entry = price
            
            # LONG: Price breaks above upper Bollinger Band with volume confirmation
            if (price > upper_band_val) and (volume_val > 1.5 * volume_sma_val):
                signals[i] = 0.25
                position = 1
                highest_since_entry = price
                continue
            
            # SHORT: Price breaks below lower Bollinger Band with volume confirmation
            elif (price < lower_band_val) and (volume_val > 1.5 * volume_sma_val):
                signals[i] = -0.25
                position = -1
                lowest_since_entry = price
                continue
        
        # Hold current position and update trailing stop levels
        if position == 1:
            highest_since_entry = max(highest_since_entry, price)
            signals[i] = 0.25
        elif position == -1:
            lowest_since_entry = min(lowest_since_entry, price)
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_BollingerBreakout_Volume_Confirmation"
timeframe = "4h"
leverage = 1.0