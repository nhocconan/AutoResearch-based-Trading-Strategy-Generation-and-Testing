#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h volume confirmation and chop regime filter
# Uses Donchian channel breakouts for trend capture, confirmed by 12h volume spike
# and filtered by 12h chop regime to avoid false signals in ranging markets.
# Works in both bull and bear markets by taking breakouts aligned with higher timeframe structure.
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
    
    # === 4h Donchian Channel (20-period) ===
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # === 12h Chop regime filter (Ehler's Chop Index) ===
    tr_12h = np.maximum(np.abs(high_12h - low_12h),
                        np.maximum(np.abs(high_12h - np.roll(close_12h, 1)),
                                   np.abs(low_12h - np.roll(close_12h, 1))))
    tr_12h[0] = high_12h[0] - low_12h[0]  # First value
    atr_sum_14 = pd.Series(tr_12h).rolling(window=14, min_periods=14).sum().values
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
        if (np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or
            np.isnan(chop_aligned[i]) or
            np.isnan(vol_spike_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        donch_high = donchian_high[i]
        donch_low = donchian_low[i]
        chop = chop_aligned[i]
        vol_conf = vol_spike_aligned[i]
        
        # === STOPLOSS LOGIC (ATR-based) ===
        if position == 1:  # Long position
            atr_4h = np.maximum(np.abs(high_4h - low_4h),
                                np.maximum(np.abs(high_4h - np.roll(close_4h, 1)),
                                           np.abs(low_4h - np.roll(close_4h, 1))))
            atr_4h[0] = high_4h[0] - low_4h[0]
            atr_ma = pd.Series(atr_4h).rolling(window=14, min_periods=14).mean().values
            atr_aligned = align_htf_to_ltf(prices, df_4h, atr_ma)
            atr_val = atr_aligned[i]
            if price < entry_price - 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            atr_4h = np.maximum(np.abs(high_4h - low_4h),
                                np.maximum(np.abs(high_4h - np.roll(close_4h, 1)),
                                           np.abs(low_4h - np.roll(close_4h, 1))))
            atr_4h[0] = high_4h[0] - low_4h[0]
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
            # Exit when price breaks below Donchian low
            if price < donch_low:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            # Exit when price breaks above Donchian high
            if price > donch_high:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Require trending regime (Chop < 38.2) and volume spike
            if chop < 38.2 and vol_conf:
                # Go long when price breaks above Donchian high with volume
                if price > donch_high:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                    continue
                # Go short when price breaks below Donchian low with volume
                elif price < donch_low:
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

name = "4h_Donchian20_12hVolumeSpike_ChopFilter"
timeframe = "4h"
leverage = 1.0