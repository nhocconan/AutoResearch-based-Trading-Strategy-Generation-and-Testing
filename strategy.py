#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + 1d volume spike + chop regime filter
# Long when price breaks above Donchian(20) high AND 1d volume > 2.0x 20-median AND chop > 61.8 (range)
# Short when price breaks below Donchian(20) low AND 1d volume > 2.0x 20-median AND chop > 61.8 (range)
# Exit when price crosses Donchian midpoint OR ATR stop (2.0)
# Position size 0.25 to balance capture and fee drag. Target: 50-150 total trades over 4 years.

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
    
    # Get 1d data once before loop for volume spike and chop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # === 1d Indicators ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 1d volume median (20-period) for spike detection
    vol_median_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).median().values
    vol_median_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_median_20_1d)
    
    # Chop index (14-period) for regime filter
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr3 = np.abs(low_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_14.sum() / (highest_high_14 - lowest_low_14)) / np.log10(14)
    chop = pd.Series(chop).rolling(window=14, min_periods=14).mean().values
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # ATR for stoploss (14-period) on primary timeframe
    tr1_ltf = high - low
    tr2_ltf = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr3_ltf = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr_ltf = np.maximum(tr1_ltf, np.maximum(tr2_ltf, tr3_ltf))
    atr_14_ltf = pd.Series(tr_ltf).rolling(window=14, min_periods=14).mean().values
    
    # Donchian channels (20-period) on primary timeframe
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_high_20 + lowest_low_20) / 2.0
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(60, 20, 14)  # 1d volume median, Donchian, chop
    
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
        if (np.isnan(vol_median_20_1d_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(atr_14_ltf[i]) or np.isnan(highest_high_20[i]) or 
            np.isnan(lowest_low_20[i]) or np.isnan(donchian_mid[i])):
            signals[i] = 0.0
            continue
        
        # Current 1d volume (aligned)
        vol_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
        if np.isnan(vol_1d_aligned[i]):
            signals[i] = 0.0
            continue
            
        # Volume filter: current 1d volume > 2.0x 20-period 1d volume median
        vol_threshold = vol_median_20_1d_aligned[i] * 2.0
        vol_confirm = vol_1d_aligned[i] > vol_threshold
        
        # Chop filter: chop > 61.8 indicates ranging market (good for mean reversion/breakouts in range)
        chop_filter = chop_aligned[i] > 61.8
        
        # Donchian breakout conditions
        price = close[i]
        upper_channel = highest_high_20[i]
        lower_channel = lowest_low_20[i]
        mid_channel = donchian_mid[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        if position == 1:  # long position
            # Exit on Donchian midpoint cross or ATR stoploss
            if price <= mid_channel or price <= entry_price - 2.0 * atr_14_ltf[i]:
                exit_signal = True
        elif position == -1:  # short position
            # Exit on Donchian midpoint cross or ATR stoploss
            if price >= mid_channel or price >= entry_price + 2.0 * atr_14_ltf[i]:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG CONDITIONS
            # Price breaks above Donchian upper channel AND volume confirmation AND chop regime
            if price > upper_channel and vol_confirm and chop_filter:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT CONDITIONS
            # Price breaks below Donchian lower channel AND volume confirmation AND chop regime
            elif price < lower_channel and vol_confirm and chop_filter:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = 0.0  # maintain position
    
    return signals

name = "12h_Donchian20_1dVolSpike_ChopFilter_v1"
timeframe = "12h"
leverage = 1.0