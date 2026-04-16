#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R1/S1 breakout with 1d volume confirmation and chop regime filter.
# Long when price breaks above Camarilla R1 (1d) AND 1d volume > 1.5x 20-period average AND chop < 61.8 (trending).
# Short when price breaks below Camarilla S1 (1d) AND 1d volume > 1.5x 20-period average AND chop < 61.8 (trending).
# Exit on opposite Camarilla level touch (S1 for long, R1 for short) or ATR-based stoploss (1.5*ATR).
# Uses discrete position size 0.25. Designed for 12h timeframe to capture multi-day trends with minimal fees.
# Works in both bull and bear markets by requiring volume confirmation and trending regime (chop filter).

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 12h Indicators: ATR for stoploss ===
    tr1 = pd.Series(high).diff()
    tr2 = pd.Series(low).diff().abs()
    tr3 = pd.Series(close).shift(1).diff().abs()
    tr_12h = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_12h_raw = pd.Series(tr_12h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # === 1d Indicators: Camarilla levels (R1, S1), Volume, Choppiness ===
    df_1d = get_htf_data(prices, '1d')
    # Camarilla levels: based on previous day's range
    # R1 = close + 1.1*(high-low)/12
    # S1 = close - 1.1*(high-low)/12
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    rng_1d = high_1d - low_1d
    camarilla_r1 = close_1d + 1.1 * rng_1d / 12
    camarilla_s1 = close_1d - 1.1 * rng_1d / 12
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Volume spike: current 1d volume > 1.5x 20-period average
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    volume_spike = vol_1d > (1.5 * vol_ma_1d_aligned)
    
    # Choppiness Index: CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    # We want trending regime: CHOP < 61.8
    # CHOP = 100 * log10(sum(ATR(1), n) / (max(high, n) - min(low, n))) / log10(n)
    # Simplified: use rolling max/min and ATR sum
    atr_1d_raw = pd.Series(tr_12h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values  # approximate 1d ATR from 12h TR
    # Actually compute proper 1d ATR from 1d data
    tr1_1d = pd.Series(high_1d).diff()
    tr2_1d = pd.Series(low_1d).diff().abs()
    tr3_1d = pd.Series(close_1d).shift(1).diff().abs()
    tr_1d = pd.concat([tr1_1d, tr2_1d, tr3_1d], axis=1).max(axis=1)
    atr_1d = pd.Series(tr_1d).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    atr_sum_1d = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    max_high_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop_denom = max_high_1d - min_low_1d
    chop_denom_safe = np.where(chop_denom == 0, 1e-10, chop_denom)
    chop = 100 * np.log10(atr_sum_1d / chop_denom_safe) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    chop_filter = chop_aligned < 61.8  # trending regime
    
    # Session filter: 08-20 UTC (optional, can be removed if too restrictive)
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 50
    
    # Track position state and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(volume_spike[i]) or np.isnan(chop_filter[i]) or np.isnan(atr_12h_raw[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        vol_spike = volume_spike[i]
        chop_trend = chop_filter[i]
        atr_val = atr_12h_raw[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price touches or breaks below Camarilla S1 (support)
            if price <= camarilla_s1_aligned[i]:
                exit_signal = True
            # ATR-based stoploss: 1.5*ATR below entry
            elif price < entry_price - 1.5 * atr_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price touches or breaks above Camarilla R1 (resistance)
            if price >= camarilla_r1_aligned[i]:
                exit_signal = True
            # ATR-based stoploss: 1.5*ATR above entry
            elif price > entry_price + 1.5 * atr_val:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above Camarilla R1 AND volume spike AND trending regime
            if (price > camarilla_r1_aligned[i] and vol_spike and chop_trend):
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: Price breaks below Camarilla S1 AND volume spike AND trending regime
            elif (price < camarilla_s1_aligned[i] and vol_spike and chop_trend):
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_Volume_Chop_V1"
timeframe = "12h"
leverage = 1.0