#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Bollinger Band breakout with 1d volume spike and chop regime filter
# Long when price breaks above BB upper (20,2) AND 1d volume > 1.5x 20-period median AND chop > 61.8 (range)
# Short when price breaks below BB lower (20,2) AND 1d volume > 1.5x 20-period median AND chop > 61.8
# Exit when price reverts to BB middle (20-period SMA) or opposite BB band touched
# Uses discrete position size 0.25 to limit fee drag. Target: 50-150 total trades over 4 years.
# Bollinger Bands work well in ranging markets (chop > 61.8) and volume spikes confirm breakout validity.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) for filter
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data once before loop for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === 1d Indicators: Volume median (20-period) ===
    volume_1d = df_1d['volume'].values
    vol_median_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).median().values
    vol_median_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_median_20_1d)
    
    # Get 12h data for Bollinger Bands and chop filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # === 12h Indicators: Bollinger Bands (20,2) ===
    close_12h = df_12h['close'].values
    sma_20_12h = pd.Series(close_12h).rolling(window=20, min_periods=20).mean().values
    std_20_12h = pd.Series(close_12h).rolling(window=20, min_periods=20).std().values
    bb_upper = sma_20_12h + 2 * std_20_12h
    bb_lower = sma_20_12h - 2 * std_20_12h
    bb_middle = sma_20_12h
    
    # Align BB levels to primary timeframe
    bb_upper_aligned = align_htf_to_ltf(prices, df_12h, bb_upper)
    bb_lower_aligned = align_htf_to_ltf(prices, df_12h, bb_lower)
    bb_middle_aligned = align_htf_to_ltf(prices, df_12h, bb_middle)
    
    # === 12h Indicators: Chopiness Index (14-period) for regime filter ===
    # Chop = 100 * log10(sum(TR)/ (N * (max(high)-min(low)))) / log10(N)
    tr1 = high - low
    tr2 = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr3 = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # For 12h data
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    tr1_12h = high_12h - low_12h
    tr2_12h = np.abs(high_12h - np.concatenate([[close_12h[0]], close_12h[:-1]]))
    tr3_12h = np.abs(low_12h - np.concatenate([[close_12h[0]], close_12h[:-1]]))
    tr_12h = np.maximum(tr1_12h, np.maximum(tr2_12h, tr3_12h))
    
    atr_sum_12h = pd.Series(tr_12h).rolling(window=14, min_periods=14).sum().values
    max_high_12h = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    min_low_12h = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    range_12h = max_high_12h - min_low_12h
    
    # Avoid division by zero
    chop_12h = np.where(range_12h > 0, 
                        100 * np.log10(atr_sum_12h / (14 * range_12h)) / np.log10(14), 
                        50.0)  # neutral when range=0
    
    chop_aligned = align_htf_to_ltf(prices, df_12h, chop_12h)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(30, 20, 20, 14)  # 1d volume, 12h BB/chop
    
    # Track position state for exits
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            if position != 0:
                position = 0  # force flat outside session
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(bb_upper_aligned[i]) or np.isnan(bb_lower_aligned[i]) or 
            np.isnan(bb_middle_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(vol_median_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current 1d volume (aligned)
        vol_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
        if np.isnan(vol_1d_aligned[i]):
            signals[i] = 0.0
            continue
            
        # Volume filter: current 1d volume > 1.5x 20-period 1d volume median
        vol_threshold = vol_median_20_1d_aligned[i] * 1.5
        vol_confirm = vol_1d_aligned[i] > vol_threshold
        
        # Chop filter: chop > 61.8 indicates ranging market (good for mean reversion)
        chop_filter = chop_aligned[i] > 61.8
        
        # Price levels
        price = close[i]
        upper = bb_upper_aligned[i]
        lower = bb_lower_aligned[i]
        middle = bb_middle_aligned[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        if position == 1:  # long position
            # Exit when price reverts to middle OR touches lower band
            if price <= middle or price <= lower:
                exit_signal = True
        elif position == -1:  # short position
            # Exit when price reverts to middle OR touches upper band
            if price >= middle or price >= upper:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG CONDITIONS
            # Price breaks above BB upper AND volume confirmation AND chop filter (ranging market)
            if price > upper and vol_confirm and chop_filter:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT CONDITIONS
            # Price breaks below BB lower AND volume confirmation AND chop filter (ranging market)
            elif price < lower and vol_confirm and chop_filter:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25  # maintain position
    
    return signals

name = "12h_BB20_1dVolumeSpike_ChopFilter_v1"
timeframe = "12h"
leverage = 1.0