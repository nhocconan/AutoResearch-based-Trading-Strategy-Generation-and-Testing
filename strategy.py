#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h chart with 1d/1w pivot confluence and volume confirmation
# Uses 1d and 1w Camarilla pivot levels as dynamic support/resistance
# Long when price breaks above R1 with volume in uptrend (above 1w EMA50)
# Short when price breaks below S1 with volume in downtrend (below 1w EMA50)
# Pivot levels provide institutional reference points that work in both bull/bear markets
# Volume filter ensures breakouts have conviction
# Target: 15-35 trades/year (60-140 over 4 years) to avoid fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d data for Camarilla pivots ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === 1w data for trend filter ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # === Calculate 1d Camarilla pivot levels ===
    # Camarilla: R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low), etc.
    # We'll use R1/S1 for entries and R4/S4 for stop placement
    hl_range = high_1d - low_1d
    r1 = close_1d + 1.1 * hl_range / 12
    s1 = close_1d - 1.1 * hl_range / 12
    r4 = close_1d + 1.5 * hl_range
    s4 = close_1d - 1.5 * hl_range
    
    # === 1w EMA50 for trend filter ===
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # === Align all 1d/1w data to 6h chart ===
    r1_6h = align_htf_to_ltf(prices, df_1d, r1)
    s1_6h = align_htf_to_ltf(prices, df_1d, s1)
    r4_6h = align_htf_to_ltf(prices, df_1d, r4)
    s4_6h = align_htf_to_ltf(prices, df_1d, s4)
    ema_50_1w_6h = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # === 6h volume confirmation ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma_20
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 100
    
    # Track position and entry price
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(r1_6h[i]) or np.isnan(s1_6h[i]) or 
            np.isnan(r4_6h[i]) or np.isnan(s4_6h[i]) or
            np.isnan(ema_50_1w_6h[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        r1 = r1_6h[i]
        s1 = s1_6h[i]
        r4 = r4_6h[i]
        s4 = s4_6h[i]
        ema_trend = ema_50_1w_6h[i]
        vol_ratio_val = vol_ratio[i]
        
        # === STOPLOSS: Exit if price reaches opposite extreme level ===
        if position == 1:  # Long position
            if price <= s4:  # Stop if price breaks below S4
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            if price >= r4:  # Stop if price breaks above R4
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === EXIT LOGIC: Reverse on opposite signal ===
        if position == 1:  # Long position
            # Exit long if price breaks below S1 (contrary signal)
            if price < s1:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            # Exit short if price breaks above R1 (contrary signal)
            if price > r1:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Break above R1 with volume, in uptrend (above 1w EMA50)
            if (price > r1 and vol_ratio_val > 1.5 and price > ema_trend):
                signals[i] = 0.25
                position = 1
                entry_price = price
                continue
            # SHORT: Break below S1 with volume, in downtrend (below 1w EMA50)
            elif (price < s1 and vol_ratio_val > 1.5 and price < ema_trend):
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

name = "6h_Camarilla_R1S1_1wEMA50_Volume_S4Stop_v1"
timeframe = "6h"
leverage = 1.0