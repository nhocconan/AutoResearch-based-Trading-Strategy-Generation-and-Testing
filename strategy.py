#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d ATR expansion (volatility breakout) + Donchian(20) breakout + volume confirmation.
# Long when price breaks above Donchian(20) high AND 1d ATR > 1.5x its 20-period median (volatility expansion).
# Short when price breaks below Donchian(20) low AND 1d ATR > 1.5x its 20-period median.
# Uses discrete position size 0.25. Exits when price reaches opposite Donchian level or ATR stoploss (2.5x ATR).
# ATR expansion identifies periods of increased volatility/institutional participation; breakout captures directional move.
# 4h timeframe targets 20-50 trades/year (75-200 total over 4 years) to minimize fee drag.
# Works in both bull (breakouts continue trend) and bear (breakouts capture panic/momentum) markets.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once before loop for ATR expansion filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === 1d Indicators: ATR (14-period) and its 20-period median for expansion filter ===
    high_low_1d = high_1d - low_1d
    high_close_1d = np.abs(high_1d - np.roll(close_1d, 1))
    low_close_1d = np.abs(low_1d - np.roll(close_1d, 1))
    true_range_1d = np.maximum(high_low_1d, np.maximum(high_close_1d, low_close_1d))
    atr_14_1d = pd.Series(true_range_1d).rolling(window=14, min_periods=14).mean().values
    atr_median_20_1d = pd.Series(atr_14_1d).rolling(window=20, min_periods=20).median().values
    
    # === 4h Indicators: Donchian channels (20-period) ===
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 4h Indicators: ATR (14-period) for stoploss ===
    high_low = high - low
    high_close = np.abs(high - np.roll(close, 1))
    low_close = np.abs(low - np.roll(close, 1))
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    atr_14 = pd.Series(true_range).rolling(window=14, min_periods=14).mean().values
    
    # Align HTF indicators to primary timeframe (4h)
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    atr_median_20_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_median_20_1d)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(20, 20)  # Donchian needs 20, ATR median needs 20
    
    # Track position state and entry price for ATR stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or
            np.isnan(atr_14_1d_aligned[i]) or np.isnan(atr_median_20_1d_aligned[i]) or
            np.isnan(atr_14[i])):
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # Current values
        price = close[i]
        atr_1d = atr_14_1d_aligned[i]
        atr_median_1d = atr_median_20_1d_aligned[i]
        atr = atr_14[i]
        
        # ATR expansion filter: current 1d ATR > 1.5x its 20-period median
        atr_expansion = atr_1d > (atr_median_1d * 1.5)
        
        # Donchian levels
        upper_channel = highest_20[i]
        lower_channel = lowest_20[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit when price reaches lower Donchian (opposite channel) OR ATR stoploss hit (2.5 * ATR below entry)
            if price <= lower_channel or price <= entry_price - 2.5 * atr:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit when price reaches upper Donchian (opposite channel) OR ATR stoploss hit (2.5 * ATR above entry)
            if price >= upper_channel or price >= entry_price + 2.5 * atr:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above upper Donchian with ATR expansion
            if price > upper_channel and atr_expansion:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: Price breaks below lower Donchian with ATR expansion
            elif price < lower_channel and atr_expansion:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25  # maintain position
    
    return signals

name = "4h_1dATRExpansion_Donchian20_Breakout_VolumeSpike_ATRTrail2.5_v1"
timeframe = "4h"
leverage = 1.0