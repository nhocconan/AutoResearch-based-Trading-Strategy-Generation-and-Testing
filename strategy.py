#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R1/S1 breakout with 1d volume spike filter and 1w ADX regime filter.
# Long when price > R1, 1d volume > 2.0x its 20-period median, and weekly ADX > 25 (trending market).
# Short when price < S1, same volume condition, and weekly ADX > 25.
# Exit when price crosses the Camarilla pivot point (mean reversion).
# Uses discrete position size 0.25. Target: 75-200 total trades over 4 years (19-50/year).
# Combines intraday price levels with volume confirmation and trend regime filter for robustness.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once before loop for Camarilla levels and volume calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === 1d Indicators: Camarilla pivot levels (R1, S1, PP) and volume ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    vol_1d = df_1d['volume'].values
    
    # Camarilla levels calculation (based on previous day)
    # R1 = close + (high - low) * 1.1/12
    # S1 = close - (high - low) * 1.1/12
    # PP = (high + low + close) / 3
    camarilla_r1 = close_1d + (high_1d - low_1d) * 1.1 / 12
    camarilla_s1 = close_1d - (high_1d - low_1d) * 1.1 / 12
    camarilla_pp = (high_1d + low_1d + close_1d) / 3
    
    # Volume median for spike filter
    vol_median_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).median().values
    
    # Get weekly data for ADX regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # === Weekly Indicators: ADX(14) trend filter ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range calculation
    tr1 = pd.Series(high_1w - low_1w)
    tr2 = pd.Series(np.abs(high_1w - np.roll(close_1w, 1)))
    tr3 = pd.Series(np.abs(low_1w - np.roll(close_1w, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    dm_plus = pd.Series(np.where((high_1w - np.roll(high_1w, 1)) > (np.roll(low_1w, 1) - low_1w), 
                                 np.maximum(high_1w - np.roll(high_1w, 1), 0), 0))
    dm_minus = pd.Series(np.where((np.roll(low_1w, 1) - low_1w) > (high_1w - np.roll(high_1w, 1)), 
                                  np.maximum(np.roll(low_1w, 1) - low_1w, 0), 0))
    
    # Smoothed values
    atr_1w = tr.ewm(span=14, adjust=False, min_periods=14).mean()
    dm_plus_smooth = dm_plus.ewm(span=14, adjust=False, min_periods=14).mean()
    dm_minus_smooth = dm_minus.ewm(span=14, adjust=False, min_periods=14).mean()
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / atr_1w
    di_minus = 100 * dm_minus_smooth / atr_1w
    
    # ADX calculation
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx_14 = dx.ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align all indicators to primary timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp)
    vol_median_aligned = align_htf_to_ltf(prices, df_1d, vol_median_20)
    adx_14_aligned = align_htf_to_ltf(prices, df_1w, adx_14)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(30, 20, 30)  # 1d Camarilla, 1d volume median, weekly ADX
    
    # Track position state for exits
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(camarilla_pp_aligned[i]) or np.isnan(vol_median_aligned[i]) or 
            np.isnan(adx_14_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current values (aligned)
        r1 = camarilla_r1_aligned[i]
        s1 = camarilla_s1_aligned[i]
        pp = camarilla_pp_aligned[i]
        vol_median = vol_median_aligned[i]
        adx = adx_14_aligned[i]
        
        # Price levels
        price = close[i]
        vol_current = volume[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        if position == 1:  # long position
            # Exit when price crosses below Camarilla pivot point (mean reversion)
            if price < pp:
                exit_signal = True
        elif position == -1:  # short position
            # Exit when price crosses above Camarilla pivot point (mean reversion)
            if price > pp:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Volume spike filter: current volume > 2.0x its 20-period median
            vol_spike = vol_current > (vol_median * 2.0)
            
            # Regime filter: weekly ADX > 25 (trending market)
            trending_regime = adx > 25
            
            # LONG CONDITIONS
            # Price breaks above Camarilla R1 AND volume spike AND trending regime
            if price > r1 and vol_spike and trending_regime:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT CONDITIONS
            # Price breaks below Camarilla S1 AND volume spike AND trending regime
            elif price < s1 and vol_spike and trending_regime:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25  # maintain position
    
    return signals

name = "4h_Camarilla_R1S1_1dVolumeSpike2.0x_1wADX25_v1"
timeframe = "4h"
leverage = 1.0