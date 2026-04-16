#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla Pivot Breakout with 1d Volume Spike and Choppiness Filter
# Uses Camarilla R1/S1 levels from 1d timeframe for breakout entries, filtered by
# 12h volume > 2x average and 1d choppiness index < 40 (trending market).
# Works in both bull/bear markets by trading breakouts in direction of 1d trend.
# Target: 80-120 total trades over 4 years (20-30/year) with selective entries.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d data (HTF for Camarilla calculation and filters) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # === 12h data (primary timeframe for execution) ===
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # === Camarilla Pivot Levels from 1d ===
    # Pivot = (H + L + C) / 3
    # R1 = C + (H - L) * 1.1 / 12
    # S1 = C - (H - L) * 1.1 / 12
    # R4 = C + (H - L) * 1.1 / 2
    # S4 = C - (H - L) * 1.1 / 2
    # We use R1/S1 for breakout, R4/S4 as stronger breakout confirmation
    
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    r1_1d = close_1d + (high_1d - low_1d) * 1.1 / 12.0
    s1_1d = close_1d - (high_1d - low_1d) * 1.1 / 12.0
    r4_1d = close_1d + (high_1d - low_1d) * 1.1 / 2.0
    s4_1d = close_1d - (high_1d - low_1d) * 1.1 / 2.0
    
    # Align Camarilla levels to 12h timeframe (completed 1d bar only)
    pivot_12h = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_12h = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_12h = align_htf_to_ltf(prices, df_1d, s1_1d)
    r4_12h = align_htf_to_ltf(prices, df_1d, r4_1d)
    s4_12h = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # === 1d Choppiness Index (CHOP) for regime filter ===
    # CHOP = 100 * log10(sum(ATR(14)) / log10(range)) / log10(14)
    # CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    # We want trending market: CHOP < 40
    
    # Calculate True Range
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # align with index 0
    
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Highest high and lowest low over 14 periods
    hh_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    range_14 = hh_14 - ll_14
    
    # Choppiness Index
    chop = np.full_like(close_1d, np.nan)
    mask = (atr_14 > 0) & (range_14 > 0) & ~np.isnan(atr_14) & ~np.isnan(range_14)
    chop[mask] = 100 * np.log10(np.sum(atr_14) / np.log10(14)) / np.log10(range_14[mask]) if np.sum(mask) > 0 else np.nan
    # Simplified calculation for performance
    chop_sum = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(chop_sum / np.log10(14)) / np.log10(range_14)
    chop = np.where((chop_sum > 0) & (range_14 > 0), chop, np.nan)
    
    # Align CHOP to 12h timeframe
    chop_12h = align_htf_to_ltf(prices, df_1d, chop)
    
    # === 12h Volume Spike Detection ===
    vol_ma_20 = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_12h > (2.0 * vol_ma_20)  # Require strong volume spike
    
    # === 1d Trend Filter (EMA50) ===
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 100
    
    # Track position and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(pivot_12h[i]) or np.isnan(r1_12h[i]) or np.isnan(s1_12h[i]) or
            np.isnan(r4_12h[i]) or np.isnan(s4_12h[i]) or np.isnan(chop_12h[i]) or
            np.isnan(vol_ma_20[i]) or np.isnan(ema50_12h[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        vol_spike_val = vol_spike[i]
        chop_val = chop_12h[i]
        ema50 = ema50_12h[i]
        
        # === STOPLOSS LOGIC ===
        if position == 1:  # Long position
            atr_12h = np.abs(high_12h - low_12h)
            atr_ma = pd.Series(atr_12h).rolling(window=14, min_periods=14).mean().values
            atr_aligned = align_htf_to_ltf(prices, df_12h, atr_ma)
            atr_val = atr_aligned[i]
            if price < entry_price - 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            atr_12h = np.abs(high_12h - low_12h)
            atr_ma = pd.Series(atr_12h).rolling(window=14, min_periods=14).mean().values
            atr_aligned = align_htf_to_ltf(prices, df_12h, atr_ma)
            atr_val = atr_aligned[i]
            if price > entry_price + 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when price closes below S1 (stronger: below S4)
            if price < s1_12h[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            # Exit when price closes above R1 (stronger: above R4)
            if price > r1_12h[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Require volume spike, trending market (CHOP < 40), and price on correct side of EMA50
            if vol_spike_val and (chop_val < 40):
                # Go long when price breaks above R1 and above EMA50 (uptrend)
                if price > r1_12h[i] and price > ema50:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                    continue
                # Go short when price breaks below S1 and below EMA50 (downtrend)
                elif price < s1_12h[i] and price < ema50:
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

name = "12h_Camarilla_R1S1_VolumeSpike_CHOPFilter"
timeframe = "12h"
leverage = 1.0