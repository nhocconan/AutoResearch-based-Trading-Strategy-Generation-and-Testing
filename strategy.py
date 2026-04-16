#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot R1/S1 breakout with 1d volume confirmation and chop regime filter.
# Long when price breaks above Camarilla R1 (1d) AND volume > 1.3x 20-period average AND CHOP(14) < 61.8 (trending).
# Short when price breaks below Camarilla S1 (1d) AND volume > 1.3x 20-period average AND CHOP(14) < 61.8 (trending).
# Exit when price returns to Camarilla pivot point (PP) or ATR(14) > ATR(50) * 1.5 (expanding volatility).
# Uses discrete position size 0.25. Camarilla provides intraday structure, 1d volume confirms breakout strength,
# chop filter avoids range-bound false breakouts, ATR regime filter exits during high volatility.
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once before loop for Camarilla pivot levels and volume MA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # === 1d Indicators: Camarilla pivot levels (R1, S1, PP) ===
    # Camarilla formulas: PP = (H+L+C)/3, R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    pivot_pp = (high_1d + low_1d + close_1d) / 3.0
    camarilla_r1 = close_1d + (high_1d - low_1d) * 1.1 / 12.0
    camarilla_s1 = close_1d - (high_1d - low_1d) * 1.1 / 12.0
    
    # Align Camarilla levels to 12h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1d, pivot_pp)
    r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # 1d volume moving average (20-period) for confirmation
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Get 12h data for CHOP and ATR calculations
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # === 12h Indicators: CHOP(14) for regime filter ===
    # True Range
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(14)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Highest high and lowest low over 14 periods
    hh_14 = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    
    # CHOP = 100 * log10(sum(ATR14)/ (HH14 - LL14)) / log10(14)
    sum_atr_14 = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    hh_ll_diff = hh_14 - ll_14
    # Avoid division by zero
    hh_ll_diff = np.where(hh_ll_diff == 0, 1e-10, hh_ll_diff)
    chop = 100 * np.log10(sum_atr_14 / hh_ll_diff) / np.log10(14)
    
    # Align CHOP to 12h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_12h, chop)
    
    # ATR(14) and ATR(50) for volatility regime filter
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    
    # Align ATR values to 12h timeframe
    atr_14_aligned = align_htf_to_ltf(prices, df_12h, atr_14)
    atr_50_aligned = align_htf_to_ltf(prices, df_12h, atr_50)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(vol_ma_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(atr_14_aligned[i]) or np.isnan(atr_50_aligned[i])):
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # Current values
        pp_val = pp_aligned[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        vol_ma_val = vol_ma_aligned[i]
        chop_val = chop_aligned[i]
        atr_14_val = atr_14_aligned[i]
        atr_50_val = atr_50_aligned[i]
        price = close[i]
        vol = volume[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price returns to pivot point or volatility expands significantly
            if price <= pp_val or atr_14_val > atr_50_val * 1.5:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price returns to pivot point or volatility expands significantly
            if price >= pp_val or atr_14_val > atr_50_val * 1.5:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Volume filter: volume > 1.3x 20-period average
            vol_filter = vol > 1.3 * vol_ma_val
            
            # Regime filter: CHOP < 61.8 indicates trending market (avoid range-bound)
            regime_filter = chop_val < 61.8
            
            # LONG: price breaks above Camarilla R1 with volume and regime confirmation
            if price > r1_val and vol_filter and regime_filter:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: price breaks below Camarilla S1 with volume and regime confirmation
            elif price < s1_val and vol_filter and regime_filter:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "12h_CamarillaR1S1_1dVol_ChopRegime_V1"
timeframe = "12h"
leverage = 1.0