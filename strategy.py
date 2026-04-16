#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian breakout with 1d volume confirmation and 1w trend filter.
# Uses weekly trend to determine direction: only take long breakouts in weekly uptrend,
# short breakouts in weekly downtrend. Volume > 1.3x average confirms breakout strength.
# Designed to work in both bull (buy dips in uptrend) and bear (sell rallies in downtrend).
# Target: 50-150 total trades over 4 years = 12-37/year.

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
    
    # === 1d data (volume confirmation) ===
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    
    # === 1w data (trend filter) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # === 6h Donchian Channel (20) ===
    highest_high = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    
    # === 1d volume ratio for confirmation ===
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1d = volume_1d / vol_ma_20_1d
    
    # === 1w EMA50 for trend filter ===
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align higher timeframe data to 6s timeframe
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 50
    
    # Track position and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_ratio_1d_aligned[i]) or np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        hh = highest_high[i]
        ll = lowest_low[i]
        vol_ratio = vol_ratio_1d_aligned[i]
        ema_50 = ema_50_1w_aligned[i]
        
        # === STOPLOSS LOGIC ===
        if position == 1:  # Long position
            if price < ll:  # Stop at Donchian low
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            if price > hh:  # Stop at Donchian high
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when price retouches Donchian middle or trend weakens
            middle = (hh + ll) / 2
            if price <= middle:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            # Exit when price retouches Donchian middle or trend weakens
            middle = (hh + ll) / 2
            if price >= middle:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Determine trend direction from 1w EMA50
            if close_1w[-1] > ema_50 if len(close_1w) > 0 else False:  # Simplified: use current close vs EMA
                # Weekly uptrend - look for long breakouts
                if price > hh and vol_ratio > 1.3:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                    continue
            else:
                # Weekly downtrend - look for short breakouts
                if price < ll and vol_ratio > 1.3:
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

name = "6h_Donchian20_1dVolConfirm_1wTrend_v1"
timeframe = "6h"
leverage = 1.0