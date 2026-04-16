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
    
    # === 6h data (primary) ===
    df_6h = get_htf_data(prices, '6h')
    close_6h = df_6h['close'].values
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    volume_6h = df_6h['volume'].values
    
    # === 1d data (HTF for ATR and range) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # === ATR for volatility-based entry/exit ===
    def calculate_atr(high, low, close, period):
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(np.maximum(tr1, tr2), tr3)
        tr = np.concatenate([[np.nan], tr])
        
        atr = np.full_like(tr, np.nan)
        if len(tr) >= period:
            atr[period-1] = np.nanmean(tr[1:period])
            for i in range(period, len(tr)):
                atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        return atr
    
    atr_1d = calculate_atr(high_1d, low_1d, close_1d, 14)
    
    # === Daily range for breakout levels ===
    daily_range = high_1d - low_1d
    range_ma = pd.Series(daily_range).rolling(window=20, min_periods=20).mean().values
    
    # === 6h volume filter ===
    vol_ma_20_6h = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    vol_ratio_6h = volume_6h / vol_ma_20_6h
    
    # Align HTF data to 6h timeframe
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    range_ma_aligned = align_htf_to_ltf(prices, df_1d, range_ma)
    vol_ratio_6h_aligned = align_htf_to_ltf(prices, df_6h, vol_ratio_6h)
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 50
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(atr_1d_aligned[i]) or 
            np.isnan(range_ma_aligned[i]) or 
            np.isnan(vol_ratio_6h_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        atr = atr_1d_aligned[i]
        range_ma_val = range_ma_aligned[i]
        vol_ratio = vol_ratio_6h_aligned[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit: ATR-based trailing stop or target
            if price <= entry_price - 1.5 * atr:  # Stop loss
                signals[i] = 0.0
                position = 0
                continue
            elif price >= entry_price + 3.0 * atr:  # Take profit
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit: ATR-based trailing stop or target
            if price >= entry_price + 1.5 * atr:  # Stop loss
                signals[i] = 0.0
                position = 0
                continue
            elif price <= entry_price - 3.0 * atr:  # Take profit
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Volatility breakout with volume confirmation
            # Long: break above 6h high + volatility expansion + volume
            if i >= 1 and price > high_6h[i-1] and atr > 1.2 * range_ma_val and vol_ratio > 1.8:
                signals[i] = 0.25
                position = 1
                entry_price = price
                continue
            # Short: break below 6h low + volatility expansion + volume
            elif i >= 1 and price < low_6h[i-1] and atr > 1.2 * range_ma_val and vol_ratio > 1.8:
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

name = "6h_Volatility_Breakout_ATR_Volume"
timeframe = "6h"
leverage = 1.0