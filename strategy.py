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
    
    # === Weekly data (HTF) ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # === Daily data (primary timeframe) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # === ATR on daily for stop loss ===
    tr_1d = np.maximum(high_1d - low_1d,
                       np.maximum(np.abs(high_1d - np.roll(close_1d, 1)),
                                  np.abs(low_1d - np.roll(close_1d, 1))))
    tr_1d[0] = high_1d[0] - low_1d[0]
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # === Weekly Donchian Channel (20) for trend direction ===
    highest_20w = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    lowest_20w = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    donchian_upper_w = highest_20w
    donchian_lower_w = lowest_20w
    donchian_upper_w_aligned = align_htf_to_ltf(prices, df_1w, donchian_upper_w)
    donchian_lower_w_aligned = align_htf_to_ltf(prices, df_1w, donchian_lower_w)
    
    # === Daily Donchian Channel (10) for entry ===
    highest_10d = pd.Series(high_1d).rolling(window=10, min_periods=10).max().values
    lowest_10d = pd.Series(low_1d).rolling(window=10, min_periods=10).min().values
    donchian_upper_d = highest_10d
    donchian_lower_d = lowest_10d
    
    # === Daily volume spike detection ===
    vol_ma_10d = pd.Series(volume_1d).rolling(window=10, min_periods=10).mean().values
    vol_ratio = volume_1d / vol_ma_10d
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators have valid data
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_1d_aligned[i]) or np.isnan(donchian_upper_w_aligned[i]) or 
            np.isnan(donchian_lower_w_aligned[i]) or np.isnan(donchian_upper_d[i]) or 
            np.isnan(donchian_lower_d[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close_1d[i]
        atr_1d_val = atr_1d_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when price closes below daily Donchian lower OR stops loss hit
            if (price < donchian_lower_d[i]) or (price < entry_price - 2.0 * atr_1d_val):
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when price closes above daily Donchian upper OR stops loss hit
            if (price > donchian_upper_d[i]) or (price > entry_price + 2.0 * atr_1d_val):
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price above weekly Donchian upper AND breaks above daily Donchian upper AND volume spike
            if (price > donchian_upper_w_aligned[i]) and (price > donchian_upper_d[i]) and (vol_ratio_val > 2.0):
                signals[i] = 0.25
                position = 1
                entry_price = price
                continue
            
            # SHORT: Price below weekly Donchian lower AND breaks below daily Donchian lower AND volume spike
            elif (price < donchian_lower_w_aligned[i]) and (price < donchian_lower_d[i]) and (vol_ratio_val > 2.0):
                signals[i] = -0.25
                position = -1
                entry_price = price
                continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "1d_WeeklyDonchian_Filter_DailyBreakout_Volume"
timeframe = "1d"
leverage = 1.0