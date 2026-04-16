#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian channel breakout with 12h pivot point bias and volume confirmation.
# In bull markets: buy breakouts above upper Donchian(20) when 12h pivot bias is bullish (price > pivot).
# In bear markets: sell breakdowns below lower Donchian(20) when 12h pivot bias is bearish (price < pivot).
# Volume > 1.5x average confirms breakout strength. Position size 0.25 for risk control.
# Uses price channels for structure and pivot bias for trend filtering to work in both regimes.

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
    
    # === 12h data (higher timeframe for pivot bias) ===
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # === 6h Donchian Channel (20) ===
    highest_high = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    upper_channel = highest_high
    lower_channel = lowest_low
    
    # === 12h Pivot Points (standard calculation) ===
    # Pivot = (High + Low + Close) / 3
    pivot_12h = (high_12h + low_12h + close_12h) / 3.0
    # Support 1 = (2 * Pivot) - High
    s1_12h = (2 * pivot_12h) - high_12h
    # Resistance 1 = (2 * Pivot) - Low
    r1_12h = (2 * pivot_12h) - low_12h
    
    # Align 12h pivot levels to 6s timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_12h, pivot_12h)
    s1_aligned = align_htf_to_ltf(prices, df_12h, s1_12h)
    r1_aligned = align_htf_to_ltf(prices, df_12h, r1_12h)
    
    # Pivot bias: bullish if price > pivot, bearish if price < pivot
    # We'll use the close price for bias determination
    bias_bullish = close_12h > pivot_12h
    bias_bearish = close_12h < pivot_12h
    bias_bullish_aligned = align_htf_to_ltf(prices, df_12h, bias_bullish.astype(float))
    bias_bearish_aligned = align_htf_to_ltf(prices, df_12h, bias_bearish.astype(float))
    
    # === 6h volume ratio for confirmation ===
    vol_ma_10_6h = pd.Series(volume_6h).rolling(window=10, min_periods=10).mean().values
    vol_ratio_6h = volume_6h / vol_ma_10_6h
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 100
    
    # Track position and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or 
            np.isnan(pivot_aligned[i]) or np.isnan(bias_bullish_aligned[i]) or
            np.isnan(bias_bearish_aligned[i]) or np.isnan(vol_ratio_6h[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        uc = upper_channel[i]
        lc = lower_channel[i]
        pivot = pivot_aligned[i]
        bullish_bias = bias_bullish_aligned[i] > 0.5
        bearish_bias = bias_bearish_aligned[i] > 0.5
        vol_ratio = vol_ratio_6h[i]
        
        # === STOPLOSS LOGIC ===
        if position == 1:  # Long position
            atr_6h = np.abs(high_6h - low_6h)
            atr_ma = pd.Series(atr_6h).rolling(window=14, min_periods=14).mean().values
            atr_aligned = align_htf_to_ltf(prices, df_6h, atr_ma)
            atr_val = atr_aligned[i]
            if price < entry_price - 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            atr_6h = np.abs(high_6h - low_6h)
            atr_ma = pd.Series(atr_6h).rolling(window=14, min_periods=14).mean().values
            atr_aligned = align_htf_to_ltf(prices, df_6h, atr_ma)
            atr_val = atr_aligned[i]
            if price > entry_price + 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when price breaks below lower Donchian or bias turns bearish
            if price < lc or bearish_bias:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            # Exit when price breaks above upper Donchian or bias turns bullish
            if price > uc or bullish_bias:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Bullish breakout: price breaks above upper Donchian with bullish bias and volume
            if price > uc and bullish_bias and vol_ratio > 1.5:
                signals[i] = 0.25
                position = 1
                entry_price = price
                continue
            # Bearish breakdown: price breaks below lower Donchian with bearish bias and volume
            elif price < lc and bearish_bias and vol_ratio > 1.5:
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

name = "6h_Donchian_PivotBias_VolumeFilter_v1"
timeframe = "6h"
leverage = 1.0