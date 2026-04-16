#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot R4/S4 breakout with 1d volume spike filter and ATR trailing stop
# Long when price breaks above Camarilla R4 (1d) AND 1d volume > 2.5x 20-period median AND ATR-based trailing stop
# Short when price breaks below Camarilla S4 (1d) AND 1d volume > 2.5x 20-period median AND ATR-based trailing stop
# Exit when price reverses 2.0x ATR from extreme OR reverts to Camarilla pivot point (PP)
# Uses discrete position size 0.25 to balance capture and fee drag. Target: 75-200 total trades over 4 years.
# R4/S4 levels are stronger breakout points than R3/S3, reducing false signals while maintaining edge.

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
    
    # Get 1d data once before loop for Camarilla pivots and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === 1d Indicators: Camarilla Pivot Levels (R4, S4, PP) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla calculations
    rang = high_1d - low_1d
    camarilla_pp = (high_1d + low_1d + close_1d) / 3.0
    camarilla_r4 = camarilla_pp + rang * 1.1 / 2.0  # R4 = PP + 1.1*range/2
    camarilla_s4 = camarilla_pp - rang * 1.1 / 2.0  # S4 = PP - 1.1*range/2
    
    # Align Camarilla levels to 4h timeframe
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # === 1d Indicators: Volume median (20-period) ===
    volume_1d = df_1d['volume'].values
    vol_median_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).median().values
    vol_median_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_median_20_1d)
    
    # Get 4h data for ATR stoploss
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # === 4h Indicators: ATR (14-period) for stoploss ===
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.concatenate([[close_4h[0]], close_4h[:-1]]))
    tr3 = np.abs(low_4h - np.concatenate([[close_4h[0]], close_4h[:-1]]))
    tr_4h = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_4h = pd.Series(tr_4h).rolling(window=14, min_periods=14).mean().values
    atr_aligned = align_htf_to_ltf(prices, df_4h, atr_14_4h)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(30, 20, 14)  # 1d Camarilla/volume, 4h ATR
    
    # Track position state for exits
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    max_favorable_price = 0.0  # track highest price for long, lowest for short
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            if position != 0:
                position = 0  # force flat outside session
                max_favorable_price = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(camarilla_pp_aligned[i]) or np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or 
            np.isnan(atr_aligned[i]) or np.isnan(vol_median_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current 1d volume (aligned)
        vol_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
        if np.isnan(vol_1d_aligned[i]):
            signals[i] = 0.0
            continue
            
        # Volume filter: current 1d volume > 2.5x 20-period 1d volume median
        vol_threshold = vol_median_20_1d_aligned[i] * 2.5
        vol_confirm = vol_1d_aligned[i] > vol_threshold
        
        # Price levels
        price = close[i]
        pp = camarilla_pp_aligned[i]
        r4 = camarilla_r4_aligned[i]
        s4 = camarilla_s4_aligned[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        if position == 1:  # long position
            # Update max favorable price
            if price > max_favorable_price:
                max_favorable_price = price
            # Exit when price retraces 2.0x ATR from high OR reverts to pivot point
            if price <= max_favorable_price - 2.0 * atr_aligned[i] or price <= pp:
                exit_signal = True
        elif position == -1:  # short position
            # Update max favorable price (lowest price for short)
            if price < max_favorable_price or max_favorable_price == 0.0:
                max_favorable_price = price
            # Exit when price rallies 2.0x ATR from low OR reverts to pivot point
            if price >= max_favorable_price + 2.0 * atr_aligned[i] or price >= pp:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            max_favorable_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG CONDITIONS
            # Price breaks above Camarilla R4 AND volume confirmation
            if price > r4 and vol_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = price
                max_favorable_price = price
            
            # SHORT CONDITIONS
            # Price breaks below Camarilla S4 AND volume confirmation
            elif price < s4 and vol_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = price
                max_favorable_price = price
        
        else:
            signals[i] = position * 0.25  # maintain position
    
    return signals

name = "4h_CamarillaR4S4_1dVolumeSpike_v1"
timeframe = "4h"
leverage = 1.0