#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 12h pivot (R1/S1) and volume confirmation
# Uses 6h Donchian breakout for entry, filtered by 12h Camarilla pivot (R1/S1) direction
# and volume > 1.5x average. Works in trending markets by buying breakouts in uptrend
# and selling breakdowns in downtrend. Target: 80-150 total trades over 4 years (20-38/year).

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
    
    # === 12h data (for pivot levels) ===
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # === 6h Donchian channels (20-period) ===
    donch_high = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    
    # === 12h Camarilla pivot levels ===
    # Pivot = (H + L + C) / 3
    # R1 = C + (H - L) * 1.1 / 12
    # S1 = C - (H - L) * 1.1 / 12
    pivot_12h = (high_12h + low_12h + close_12h) / 3
    r1_12h = close_12h + (high_12h - low_12h) * 1.1 / 12
    s1_12h = close_12h - (high_12h - low_12h) * 1.1 / 12
    
    # Align pivot levels to 6h timeframe
    pivot_12h_aligned = align_htf_to_ltf(prices, df_12h, pivot_12h)
    r1_12h_aligned = align_htf_to_ltf(prices, df_12h, r1_12h)
    s1_12h_aligned = align_htf_to_ltf(prices, df_12h, s1_12h)
    
    # === 6h volume spike detection ===
    vol_ma_20_6h = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_6h > (1.5 * vol_ma_20_6h)
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 100
    
    # Track position and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(donch_high[i]) or 
            np.isnan(donch_low[i]) or
            np.isnan(pivot_12h_aligned[i]) or
            np.isnan(r1_12h_aligned[i]) or
            np.isnan(s1_12h_aligned[i]) or
            np.isnan(vol_ma_20_6h[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        donch_high_val = donch_high[i]
        donch_low_val = donch_low[i]
        pivot_val = pivot_12h_aligned[i]
        r1_val = r1_12h_aligned[i]
        s1_val = s1_12h_aligned[i]
        vol_spike_val = vol_spike[i]
        
        # === STOPLOSS LOGIC ===
        if position == 1:  # Long position
            atr_6h = np.abs(high_6h - low_6h)
            atr_ma = pd.Series(atr_6h).rolling(window=14, min_periods=14).mean().values
            atr_aligned = align_htf_to_ltf(prices, df_6h, atr_ma)
            atr_val = atr_aligned[i]
            if price < entry_price - 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            atr_6h = np.abs(high_6h - low_6h)
            atr_ma = pd.Series(atr_6h).rolling(window=14, min_periods=14).mean().values
            atr_aligned = align_htf_to_ltf(prices, df_6h, atr_ma)
            atr_val = atr_aligned[i]
            if price > entry_price + 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when price breaks below S1 pivot level
            if price < s1_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            # Exit when price breaks above R1 pivot level
            if price > r1_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Require volume spike
            if vol_spike_val:
                # Go long when price breaks above Donchian high AND above R1 pivot (uptrend confirmation)
                if price > donch_high_val and price > r1_val:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                    continue
                # Go short when price breaks below Donchian low AND below S1 pivot (downtrend confirmation)
                elif price < donch_low_val and price < s1_val:
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

name = "6h_Donchian20_Pivot12h_Volume_Confirmation"
timeframe = "6h"
leverage = 1.0