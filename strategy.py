#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d pivot direction filter and volume confirmation.
# Donchian breakouts capture strong directional moves. 1d pivot provides market bias (bullish/bearish).
# Volume > 2x average confirms breakout strength. Works in bull (breakouts up) and bear (breakdowns down).
# Target: 50-150 total trades over 4 years (12-37/year). Position size: 0.25.

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
    close_6h = df_6h['close'].values
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    volume_6h = df_6h['volume'].values
    
    # === 1d data (higher timeframe for pivot and trend filter) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === 1d Pivot Point (standard) ===
    # Pivot = (H + L + C) / 3
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    # Bias: above pivot = bullish, below = bearish
    bias_bullish = pivot_1d > 0  # Will be replaced with actual comparison
    bias_bearish = pivot_1d < 0  # Placeholder, will fix below
    
    # Fix: Calculate actual bias by comparing pivot to close
    bias_bullish = close_1d > pivot_1d
    bias_bearish = close_1d < pivot_1d
    
    # Align bias to 6h timeframe (use previous day's bias)
    bias_bullish_aligned = align_htf_to_ltf(prices, df_1d, bias_bullish.astype(float))
    bias_bearish_aligned = align_htf_to_ltf(prices, df_1d, bias_bearish.astype(float))
    
    # === 6h Donchian Channel (20 periods) ===
    # Upper band = highest high of last 20 periods
    # Lower band = lowest low of last 20 periods
    high_series = pd.Series(high_6h)
    low_series = pd.Series(low_6h)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Align Donchian bands to 6x frequency (they're already on 6h)
    # No alignment needed as df_6h is already 6h data
    
    # === 6h volume ratio for confirmation ===
    vol_ma_10_6h = pd.Series(volume_6h).rolling(window=10, min_periods=10).mean().values
    vol_ratio_6h = volume_6h / vol_ma_10_6h
    
    signals = np.zeros(n)
    
    # Warmup: need enough data for Donchian (20) and volume MA (10)
    warmup = 30
    
    # Track position and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(bias_bullish_aligned[i]) or np.isnan(bias_bearish_aligned[i]) or
            np.isnan(vol_ratio_6h[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        upper = donchian_upper[i]
        lower = donchian_lower[i]
        bullish_bias = bias_bullish_aligned[i] > 0.5
        bearish_bias = bias_bearish_aligned[i] > 0.5
        vol_ratio = vol_ratio_6h[i]
        
        # === STOPLOSS LOGIC ===
        if position == 1:  # Long position
            atr_6h = np.abs(high_6h - low_6h)
            atr_ma = pd.Series(atr_6h).rolling(window=14, min_periods=14).mean().values
            atr_val = atr_ma[i]
            if price < entry_price - 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            atr_6h = np.abs(high_6h - low_6h)
            atr_ma = pd.Series(atr_6h).rolling(window=14, min_periods=14).mean().values
            atr_val = atr_ma[i]
            if price > entry_price + 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when price breaks below Donchian lower or bias flips
            if price < lower or not bullish_bias:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            # Exit when price breaks above Donchian upper or bias flips
            if price > upper or not bearish_bias:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Breakout above upper band with volume and bullish bias
            if price > upper and vol_ratio > 2.0 and bullish_bias:
                signals[i] = 0.25
                position = 1
                entry_price = price
                continue
            # Breakdown below lower band with volume and bearish bias
            elif price < lower and vol_ratio > 2.0 and bearish_bias:
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

name = "6h_Donchian20_PivotBias_VolumeFilter_v1"
timeframe = "6h"
leverage = 1.0