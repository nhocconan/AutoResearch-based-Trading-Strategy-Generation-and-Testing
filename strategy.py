#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h TRIX momentum with 12h volume spike and chop regime filter
# Uses 4h primary timeframe with 12h HTF for volume confirmation and chop regime.
# TRIX (triple-smoothed EMA) filters noise and identifies sustained momentum.
# Volume spike confirms institutional participation. Chop regime avoids false signals in ranging markets.
# Works in both bull and bear markets by taking momentum breakouts aligned with higher timeframe structure.
# Target: 75-200 trades over 4 years (19-50/year) to balance statistical significance and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h data (primary timeframe) ===
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # === 12h data (HTF for volume confirmation, chop regime) ===
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # === 12h TRIX (15-period) ===
    # TRIX = EMA(EMA(EMA(close, 15), 15), 15)
    ema1 = pd.Series(close_12h).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    trix = pd.Series(ema3).pct_change() * 100  # Percentage change
    trix_values = trix.values
    
    # Align TRIX to 4h timeframe (wait for 12h bar close)
    trix_aligned = align_htf_to_ltf(prices, df_12h, trix_values)
    
    # === 12h Chop regime filter (Ehler's Chop Index) ===
    atr_12h = np.abs(high_12h - low_12h)
    atr_sum_14 = pd.Series(atr_12h).rolling(window=14, min_periods=14).sum().values
    max_high_14 = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    chop_raw = 100 * np.log10(atr_sum_14 / (max_high_14 - min_low_14 + 1e-10)) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_12h, chop_raw)
    
    # === 12h Volume confirmation ===
    vol_ma_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_12h > (2.0 * vol_ma_20_12h)
    vol_spike_aligned = align_htf_to_ltf(prices, df_12h, vol_spike)
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 100
    
    # Track position and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(trix_aligned[i]) or 
            np.isnan(chop_aligned[i]) or
            np.isnan(vol_spike_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        trix_val = trix_aligned[i]
        chop = chop_aligned[i]
        vol_conf = vol_spike_aligned[i]
        
        # === STOPLOSS LOGIC (ATR-based) ===
        if position == 1:  # Long position
            atr_4h = np.abs(high_4h - low_4h)
            atr_ma = pd.Series(atr_4h).rolling(window=14, min_periods=14).mean().values
            atr_aligned = align_htf_to_ltf(prices, df_4h, atr_ma)
            atr_val = atr_aligned[i]
            if price < entry_price - 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            atr_4h = np.abs(high_4h - low_4h)
            atr_ma = pd.Series(atr_4h).rolling(window=14, min_periods=14).mean().values
            atr_aligned = align_htf_to_ltf(prices, df_4h, atr_ma)
            atr_val = atr_aligned[i]
            if price > entry_price + 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when TRIX turns negative (momentum fading)
            if trix_val <= 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            # Exit when TRIX turns positive (momentum fading)
            if trix_val >= 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Require trending regime (Chop < 38.2) and volume spike
            if chop < 38.2 and vol_conf:
                # Go long when TRIX crosses above zero with volume
                if trix_val > 0:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                    continue
                # Go short when TRIX crosses below zero with volume
                elif trix_val < 0:
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

name = "4h_TRIX15_12hVolumeSpike_ChopFilter"
timeframe = "4h"
leverage = 1.0