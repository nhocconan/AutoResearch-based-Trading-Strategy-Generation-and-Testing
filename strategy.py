#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1h volume confirmation and 1d choppiness regime filter
# Long when price breaks above 20-period high with 1h volume > 1.5x 20-median AND 1d CHOP > 61.8 (range) → mean reversion exit at mid-channel
# Short when price breaks below 20-period low with 1h volume > 1.5x 20-median AND 1d CHOP > 61.8 (range) → mean reversion exit at mid-channel
# Exit when price reverts to mid-channel ( (upper+lower)/2 ) or ATR stop (2.0)
# Uses discrete position size 0.25 to balance capture and fee drag. Target: 100-200 total trades over 4 years.

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
    
    # Get 1d data once before loop for choppiness index
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === 1d Indicators: Choppiness Index (CHOP) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr3 = np.abs(low_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Sum of TR over 14 periods
    tr_sum_14 = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Chop = 100 * log10(sum(tr14) / (hh14 - ll14)) / log10(14)
    # Avoid division by zero
    range_14 = hh_14 - ll_14
    chop_1d = np.where(
        range_14 > 0,
        100 * np.log10(tr_sum_14 / range_14) / np.log10(14),
        50.0  # neutral when range=0
    )
    
    # Align CHOP to 4h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Get 1h data once before loop for volume confirmation
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 20:
        return np.zeros(n)
    
    # === 1h Indicators: Volume median ===
    volume_1h = df_1h['volume'].values
    vol_median_20_1h = pd.Series(volume_1h).rolling(window=20, min_periods=20).median().values
    vol_median_20_1h_aligned = align_htf_to_ltf(prices, df_1h, vol_median_20_1h)
    
    # ATR for stoploss (14-period) on 4h
    tr1_4h = high - low
    tr2_4h = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr3_4h = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr_4h = np.maximum(tr1_4h, np.maximum(tr2_4h, tr3_4h))
    atr_14_4h = pd.Series(tr_4h).rolling(window=14, min_periods=14).mean().values
    
    # Donchian channels (20-period) on 4h
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    mid_channel = (highest_20 + lowest_20) / 2.0
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(30, 20, 14)  # 1d chop, 1h volume median, 4h Donchian/ATR
    
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
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or np.isnan(mid_channel[i]) or 
            np.isnan(atr_14_4h[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(vol_median_20_1h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current 1h volume (aligned)
        vol_1h_aligned = align_htf_to_ltf(prices, df_1h, volume_1h)
        if np.isnan(vol_1h_aligned[i]):
            signals[i] = 0.0
            continue
            
        # Volume filter: current 1h volume > 1.5x 20-period 1h volume median
        vol_threshold = vol_median_20_1h_aligned[i] * 1.5
        vol_confirm = vol_1h_aligned[i] > vol_threshold
        
        # Regime filter: 1d CHOP > 61.8 (range-bound market) → mean reversion
        ranging = chop_aligned[i] > 61.8
        
        # Price levels
        price = close[i]
        upper = highest_20[i]
        lower = lowest_20[i]
        mid = mid_channel[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        if position == 1:  # long position
            # Exit when price returns to mid-channel or ATR stoploss
            if price <= mid or price <= entry_price - 2.0 * atr_14_4h[i]:
                exit_signal = True
        elif position == -1:  # short position
            # Exit when price returns to mid-channel or ATR stoploss
            if price >= mid or price >= entry_price + 2.0 * atr_14_4h[i]:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0 and ranging:
            # LONG CONDITIONS
            # Price breaks above 20-period high AND volume confirmation AND ranging market
            if price > upper and vol_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT CONDITIONS
            # Price breaks below 20-period low AND volume confirmation AND ranging market
            elif price < lower and vol_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = 0.0  # maintain position
    
    return signals

name = "4h_Donchian20_1hVol_1dChop_v1"
timeframe = "4h"
leverage = 1.0