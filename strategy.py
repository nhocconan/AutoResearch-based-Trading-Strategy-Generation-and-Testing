#!/usr/bin/env python3
"""
4h_1d_KAMA_Direction_VolumeSpike_ChopFilter
Hypothesis: 4h KAMA direction (trend filter) + 1d volume spike (>2x 20-period MA) + 4h chop regime (CHOP > 61.8 = range → mean reversion at Bollinger Bands). 
In ranging markets (CHOP > 61.8): mean revert → long at lower BB, short at upper BB on volume spike.
In trending markets (CHOP ≤ 61.8): follow KAMA direction → long if KAMA rising, short if falling.
ATR trailing stop (2.0x ATR) manages risk. Position size 0.25.
Target ~20-40 trades/year per symbol (80-160 total over 4 years).
Uses 4h primary timeframe with 1d HTF for volume spike and KAMA direction.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for volume spike and KAMA)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # === 1d Volume Spike (>2x 20-period MA) ===
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_1d > 2.0 * vol_ma_1d
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike.astype(float))
    
    # === 1d KAMA (ER=10) for trend direction ===
    # Efficiency Ratio
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.sum(np.abs(np.diff(close_1d)), axis=0)  # placeholder, will compute correctly below
    # Recompute volatility properly: sum of absolute changes over ER period
    er_period = 10
    volatility_sum = pd.Series(np.abs(np.diff(close_1d, prepend=close_1d[0]))).rolling(window=er_period, min_periods=er_period).sum().values
    change_abs = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    change_sum = pd.Series(change_abs).rolling(window=er_period, min_periods=er_period).sum().values
    er = np.divide(change_sum, volatility_sum, out=np.zeros_like(change_sum), where=volatility_sum!=0)
    # Smoothing constants
    sc = (er * (2/(2+2) - 2/(30+2)) + 2/(30+2)) ** 2  # fast=2, slow=30
    # KAMA
    kama = np.full_like(close_1d, np.nan)
    kama[er_period] = close_1d[er_period]  # seed
    for i in range(er_period+1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    kama_dir = np.diff(kama, prepend=0)  # >0 rising, <0 falling
    kama_dir_aligned = align_htf_to_ltf(prices, df_1d, kama_dir)
    
    # === 4h Indicators (primary timeframe) ===
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Bollinger Bands (20, 2.0) for mean reversion in ranging markets
    close_s = pd.Series(close_4h)
    basis = close_s.rolling(window=20, min_periods=20).mean().values
    dev = close_s.rolling(window=20, min_periods=20).std().values
    upper_band = basis + 2.0 * dev
    lower_band = basis - 2.0 * dev
    
    # Chop regime (4h) for trend/range filter
    tr1 = high_4h[1:] - low_4h[1:]
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_4h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    sum_atr = pd.Series(atr_4h).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high_4h).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low_4h).rolling(window=14, min_periods=14).min().values
    chop = 100 * (np.log10(sum_atr / (max_high - min_low + 1e-10)) / np.log10(14))
    
    # ATR (14-period) for stoploss
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values  # reuse TR from above
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(kama_dir_aligned[i]) or np.isnan(volume_spike_aligned[i]) 
            or np.isnan(chop[i]) or np.isnan(upper_band[i]) or np.isnan(lower_band[i]) 
            or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_4h[i]
        vol_spike = volume_spike_aligned[i] > 0.5  # boolean from aligned array
        kama_dir_val = kama_dir_aligned[i]
        chop_val = chop[i]
        is_ranging = chop_val > 61.8  # CHOP > 61.8 = ranging market
        
        if position == 0:
            if is_ranging:
                # In ranging market: mean revert at Bollinger Bands on volume spike
                # Long: price at lower BB + volume spike
                if price <= lower_band[i] and vol_spike:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                    highest_since_entry = price
                # Short: price at upper BB + volume spike
                elif price >= upper_band[i] and vol_spike:
                    signals[i] = -0.25
                    position = -1
                    entry_price = price
                    lowest_since_entry = price
            else:
                # In trending market: follow KAMA direction on volume spike
                # Long: KAMA rising + volume spike
                if kama_dir_val > 0 and vol_spike:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                    highest_since_entry = price
                # Short: KAMA falling + volume spike
                elif kama_dir_val < 0 and vol_spike:
                    signals[i] = -0.25
                    position = -1
                    entry_price = price
                    lowest_since_entry = price
        
        elif position == 1:
            # Update highest since entry
            highest_since_entry = max(highest_since_entry, price)
            # ATR trailing stop: exit if price drops 2.0*ATR from highest since entry
            if price < highest_since_entry - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Update lowest since entry
            lowest_since_entry = min(lowest_since_entry, price)
            # ATR trailing stop: exit if price rises 2.0*ATR from lowest since entry
            if price > lowest_since_entry + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_1d_KAMA_Direction_VolumeSpike_ChopFilter"
timeframe = "4h"
leverage = 1.0