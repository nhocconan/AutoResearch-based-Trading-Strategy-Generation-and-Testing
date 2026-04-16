#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout + 12h pivot direction + volume confirmation
# Donchian breakouts capture momentum in trending markets. 
# Pivot direction from 12h acts as trend filter: only take long breaks above 12h pivot point in uptrend,
# short breaks below pivot in downtrend. Volume confirms breakout strength.
# Works in bull markets (breakouts with volume) and bear markets (breakdowns with volume).
# Target: 50-150 total trades over 4 years (12-37/year) with disciplined entries.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
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
    
    # === 12h data (higher timeframe for pivot direction) ===
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # === 6h Donchian(20) channels ===
    donch_high = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    
    # === 12h Pivot Point (classic: (H+L+C)/3) ===
    pp_12h = (high_12h + low_12h + close_12h) / 3.0
    
    # === 6h volume ratio for confirmation ===
    vol_ma_20_6h = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    vol_ratio_6h = volume_6h / vol_ma_20_6h
    
    # Align 12h pivot to 6s timeframe
    pp_aligned = align_htf_to_ltf(prices, df_12h, pp_12h)
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 50
    
    # Track position and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(pp_aligned[i]) or np.isnan(vol_ratio_6h[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        pp_val = pp_aligned[i]
        vol_ratio = vol_ratio_6h[i]
        
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
            # Exit when price breaks below Donchian low or pivot flips bearish
            if price < donch_low[i] or close_6h[i] < pp_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            # Exit when price breaks above Donchian high or pivot flips bullish
            if price > donch_high[i] or close_6h[i] > pp_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Require volume confirmation
            if vol_ratio > 1.5:
                # Buy when price breaks above Donchian high AND above 12h pivot (bullish alignment)
                if price > donch_high[i] and close_6h[i] > pp_val:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                    continue
                # Sell when price breaks below Donchian low AND below 12h pivot (bearish alignment)
                elif price < donch_low[i] and close_6h[i] < pp_val:
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

name = "6h_Donchian20_Pivot12h_VolumeBreakout_v1"
timeframe = "6h"
leverage = 1.0