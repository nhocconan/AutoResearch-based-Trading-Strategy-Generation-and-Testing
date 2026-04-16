#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla Pivot R1/S1 breakout with 1d volume spike and 12h chop regime filter
# Uses 6h primary timeframe with 12h HTF for chop regime and 1d HTF for volume confirmation.
# Camarilla R1/S1 levels act as intraday support/resistance; breakouts with volume spike indicate strong momentum.
# Chop regime filter (from 12h) avoids false breakouts in ranging markets.
# Works in both bull and bear markets by only taking breakouts aligned with higher timeframe structure.
# Target: 50-150 trades over 4 years (12-37/year) to avoid fee drag while maintaining statistical significance.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 6h data (primary timeframe) ===
    df_6h = get_htf_data(prices, '6h')
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    volume_6h = df_6h['volume'].values
    
    # === 12h data (HTF for chop regime) ===
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # === 1d data (HTF for volume confirmation and Camarilla calculation) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # === 1d Camarilla pivot levels (R1, S1, R2, S2, R3, S3, R4, S4) ===
    # Camarilla formula: 
    # R4 = close + ((high - low) * 1.1 / 2)
    # R3 = close + ((high - low) * 1.1 / 4)
    # R2 = close + ((high - low) * 1.1 / 6)
    # R1 = close + ((high - low) * 1.1 / 12)
    # PP = (high + low + close) / 3
    # S1 = close - ((high - low) * 1.1 / 12)
    # S2 = close - ((high - low) * 1.1 / 6)
    # S3 = close - ((high - low) * 1.1 / 4)
    # S4 = close - ((high - low) * 1.1 / 2)
    rng = high_1d - low_1d
    camarilla_r1 = close_1d + (rng * 1.1 / 12)
    camarilla_s1 = close_1d - (rng * 1.1 / 12)
    camarilla_r2 = close_1d + (rng * 1.1 / 6)
    camarilla_s2 = close_1d - (rng * 1.1 / 6)
    camarilla_r3 = close_1d + (rng * 1.1 / 4)
    camarilla_s3 = close_1d - (rng * 1.1 / 4)
    camarilla_r4 = close_1d + (rng * 1.1 / 2)
    camarilla_s4 = close_1d - (rng * 1.1 / 2)
    
    # Align Camarilla levels to 6h timeframe (wait for 1d bar close)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    camarilla_r2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r2)
    camarilla_s2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s2)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # === 12h Chop regime filter (Ehler's Chop Index) ===
    # Chop = 100 * log10(sum(ATR(1)) / (max(high) - min(low))) / log10(n)
    # Chop > 61.8 = ranging, Chop < 38.2 = trending
    atr_12h = np.abs(high_12h - low_12h)
    atr_sum_14 = pd.Series(atr_12h).rolling(window=14, min_periods=14).sum().values
    max_high_14 = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    chop_raw = 100 * np.log10(atr_sum_14 / (max_high_14 - min_low_14 + 1e-10)) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_12h, chop_raw)
    
    # === 1d Volume confirmation ===
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_1d > (2.0 * vol_ma_20_1d)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike)
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 100
    
    # Track position and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(chop_aligned[i]) or 
            np.isnan(vol_spike_aligned[i]) or
            np.isnan(camarilla_r1_aligned[i]) or
            np.isnan(camarilla_s1_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        chop = chop_aligned[i]
        vol_conf = vol_spike_aligned[i]
        r1 = camarilla_r1_aligned[i]
        s1 = camarilla_s1_aligned[i]
        r2 = camarilla_r2_aligned[i]
        s2 = camarilla_s2_aligned[i]
        r3 = camarilla_r3_aligned[i]
        s3 = camarilla_s3_aligned[i]
        r4 = camarilla_r4_aligned[i]
        s4 = camarilla_s4_aligned[i]
        
        # === STOPLOSS LOGIC (ATR-based) ===
        if position == 1:  # Long position
            atr_6h = np.abs(high_6h - low_6h)
            atr_ma = pd.Series(atr_6h).rolling(window=14, min_periods=14).mean().values
            atr_aligned = align_htf_to_ltf(prices, df_6h, atr_ma)
            atr_val = atr_aligned[i]
            if price < entry_price - 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            atr_6h = np.abs(high_6h - low_6h)
            atr_ma = pd.Series(atr_6h).rolling(window=14, min_periods=14).mean().values
            atr_aligned = align_htf_to_ltf(prices, df_6h, atr_ma)
            atr_val = atr_aligned[i]
            if price > entry_price + 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when price reaches R3 (take profit) or shows weakness below R1
            if price >= r3:  # Take profit at R3
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
            elif price < r1:  # Stop loss if breaks below R1
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            # Exit when price reaches S3 (take profit) or shows strength above S1
            if price <= s3:  # Take profit at S3
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
            elif price > s1:  # Stop loss if breaks above S1
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Require trending regime (Chop < 38.2) and volume spike
            if chop < 38.2 and vol_conf:
                # Go long when price breaks above R1 with volume
                if price > r1:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                    continue
                # Go short when price breaks below S1 with volume
                elif price < s1:
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

name = "6h_Camarilla_R1S1_Breakout_VolumeSpike_ChopFilter"
timeframe = "6h"
leverage = 1.0