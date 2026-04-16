#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using weekly Donchian channel breakout with volume confirmation and ATR-based stoploss.
# Long when price breaks above weekly Donchian upper channel (20-period) with volume > 1.5x 20-period median.
# Short when price breaks below weekly Donchian lower channel (20-period) with volume > 1.5x 20-period median.
# Uses discrete position size 0.25. Exits when price crosses the weekly Donchian middle (mean reversion) or ATR stoploss hits.
# Weekly Donchian provides structure from higher timeframe, volume confirms breakout validity.
# 1d timeframe targets 7-25 trades/year (30-100 total over 4 years) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data once before loop for Donchian channels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # === Weekly Indicators: Donchian Channel (20-period) ===
    # Upper = max(high, lookback=20)
    # Lower = min(low, lookback=20)
    # Middle = (upper + lower) / 2
    lookback = 20
    high_roll_max = pd.Series(high_1w).rolling(window=lookback, min_periods=lookback).max().values
    low_roll_min = pd.Series(low_1w).rolling(window=lookback, min_periods=lookback).min().values
    upper_1w = high_roll_max
    lower_1w = low_roll_min
    middle_1w = (upper_1w + lower_1w) / 2
    
    # === Weekly Indicators: Volume Median (20-period) ===
    vol_median_20 = pd.Series(volume_1w).rolling(window=20, min_periods=20).median().values
    
    # === Daily Indicators: ATR (14-period) for stoploss ===
    high_low = high - low
    high_close = np.abs(high - np.roll(close, 1))
    low_close = np.abs(low - np.roll(close, 1))
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    atr_14 = pd.Series(true_range).rolling(window=14, min_periods=14).mean().values
    
    # Align all indicators to primary timeframe (1d)
    upper_aligned = align_htf_to_ltf(prices, df_1w, upper_1w)
    lower_aligned = align_htf_to_ltf(prices, df_1w, lower_1w)
    middle_aligned = align_htf_to_ltf(prices, df_1w, middle_1w)
    vol_median_aligned = align_htf_to_ltf(prices, df_1w, vol_median_20)
    # ATR is already on primary timeframe
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(20, 20, 14)  # Donchian lookback, volume median, ATR
    
    # Track position state and entry price for ATR stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or np.isnan(middle_aligned[i]) or
            np.isnan(vol_median_aligned[i]) or np.isnan(atr_14[i])):
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # Current values (aligned)
        price = close[i]
        vol_median = vol_median_aligned[i]
        upper = upper_aligned[i]
        lower = lower_aligned[i]
        middle = middle_aligned[i]
        atr = atr_14[i]
        
        # Get current daily volume for volume spike filter
        vol_1d_aligned = align_htf_to_ltf(prices, df_1w, volume_1w)
        current_vol_1w = vol_1d_aligned[i]
        
        # Volume spike filter: current weekly volume > 1.5x median volume
        volume_spike = current_vol_1w > (vol_median * 1.5)
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit when price reaches or falls below weekly middle (mean reversion)
            # OR ATR stoploss hit (2 * ATR below entry)
            if price <= middle or price <= entry_price - 2.0 * atr:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit when price reaches or rises above weekly middle (mean reversion)
            # OR ATR stoploss hit (2 * ATR above entry)
            if price >= middle or price >= entry_price + 2.0 * atr:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: price breaks above weekly upper channel with volume spike
            if price > upper and volume_spike:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: price breaks below weekly lower channel with volume spike
            elif price < lower and volume_spike:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25  # maintain position
    
    return signals

name = "1d_WeeklyDonchian20_VolumeSpike1.5x_ATRTrail2.0_v1"
timeframe = "1d"
leverage = 1.0