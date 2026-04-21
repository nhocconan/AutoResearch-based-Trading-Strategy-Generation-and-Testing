#!/usr/bin/env python3
"""
4h_HTF_1d_Donchian20_Breakout_Volume_ChopRegime_ATRStop_V1
Hypothesis: 4h Donchian(20) breakout with volume confirmation (>1.3x 20-period volume MA) and choppiness regime filter (CHOP>61.8 for mean reversion, CHOP<38.2 for trend following). Uses 1d HTF for choppiness calculation. ATR-based stoploss via signal=0 when price moves against position by 2.5x ATR. Designed to work in both bull and bear markets via regime adaptation. Target: 20-50 trades/year (80-200 total over 4 years) to avoid fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for choppiness regime)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # === 1d Choppiness Index (CHOP) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = tr2[0] = tr3[0] = 0  # first bar has no prior close
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(14) and sum of TR over 14 periods
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    tr_sum_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: CHOP = 100 * log10(tr_sum_14 / (atr_14 * 14)) / log10(14)
    # Avoid division by zero
    divisor = atr_14 * 14
    divisor = np.where(divisor == 0, 1e-10, divisor)
    chop_raw = 100 * np.log10(tr_sum_14 / divisor) / np.log10(14)
    chop_raw = np.where(tr_sum_14 == 0, 50, chop_raw)  # neutral when no movement
    chop_1d = chop_raw
    
    # Align choppiness to 4h timeframe
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # === 4h Indicators (primary timeframe) ===
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Donchian channels (20-period)
    dc_high_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    dc_low_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Volume MA (20-period) for spike detection
    vol_ma = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for stoploss
    tr_4h1 = np.abs(high_4h - low_4h)
    tr_4h2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr_4h3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr_4h1[0] = tr_4h2[0] = tr_4h3[0] = 0
    tr_4h = np.maximum(tr_4h1, np.maximum(tr_4h2, tr_4h3))
    atr_14_4h = pd.Series(tr_4h).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if indicators not ready
        if (np.isnan(dc_high_20[i]) or np.isnan(dc_low_20[i]) 
            or np.isnan(vol_ma[i]) or np.isnan(atr_14_4h[i])
            or np.isnan(chop_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_4h[i]
        vol = volume_4h[i]
        vol_ok = vol > 1.3 * vol_ma[i]  # volume spike confirmation
        chop = chop_1d_aligned[i]
        
        # Regime determination
        is_choppy = chop > 61.8   # range market -> mean reversion
        is_trending = chop < 38.2  # trending market -> trend follow
        
        if position == 0:
            # Long entry conditions
            long_breakout = price > dc_high_20[i]
            # In trending market: breakout long
            # In choppy market: mean reversion from lower band
            long_entry = (is_trending and long_breakout) or (is_choppy and price < dc_low_20[i])
            
            # Short entry conditions
            short_breakout = price < dc_low_20[i]
            # In trending market: breakdown short
            # In choppy market: mean reversion from upper band
            short_entry = (is_trending and short_breakout) or (is_choppy and price > dc_high_20[i])
            
            if long_entry and vol_ok:
                signals[i] = 0.25
                position = 1
                entry_price = price
            elif short_entry and vol_ok:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: stoploss or opposite signal
            stoploss_level = entry_price - 2.5 * atr_14_4h[i]
            if price < stoploss_level or price < dc_low_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: stoploss or opposite signal
            stoploss_level = entry_price + 2.5 * atr_14_4h[i]
            if price > stoploss_level or price > dc_high_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_HTF_1d_Donchian20_Breakout_Volume_ChopRegime_ATRStop_V1"
timeframe = "4h"
leverage = 1.0