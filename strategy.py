#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_VolumeSpike_RegimeFilter_V1
Hypothesis: 4h Donchian(20) breakout with volume confirmation and choppiness regime filter.
Long when price breaks above upper Donchian band in low-chop regime (trending market).
Short when price breaks below lower Donchian band in low-chop regime.
Volume confirmation (2.0x average) filters false breakouts. Uses ATR(14) stoploss (2.0x).
Designed for fewer trades (~100/year) to minimize fee drag while capturing strong trends.
Works in both bull and bear markets by requiring low choppiness (trending regime).
Timeframe: 4h, uses 1d HTF for chop filter.
Target: 75-200 total trades over 4 years = 19-50/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for choppiness filter)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # === 1d OHLC for Choppiness Index (CHOP) regime filter ===
    df_1d_high = df_1d['high'].values
    df_1d_low = df_1d['low'].values
    df_1d_close = df_1d['close'].values
    
    # True Range
    tr1 = pd.Series(df_1d_high - df_1d_low)
    tr2 = pd.Series(np.abs(df_1d_high - np.roll(df_1d_close, 1)))
    tr3 = pd.Series(np.abs(df_1d_low - np.roll(df_1d_close, 1)))
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr_1d.rolling(window=14, min_periods=14).mean().values
    
    # Sum of True Range over 14 periods
    sum_tr_14 = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
    
    # Highest High and Lowest Low over 14 periods
    hh_14 = pd.Series(df_1d_high).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(df_1d_low).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: CHOP = 100 * log10(sum_tr_14 / (hh_14 - ll_14)) / log10(14)
    # Avoid division by zero
    range_14 = hh_14 - ll_14
    range_14 = np.where(range_14 == 0, 1e-10, range_14)
    chop_1d = 100 * np.log10(sum_tr_14 / range_14) / np.log10(14)
    
    # Align 1d Chop to 4h timeframe (low chop = trending regime)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # === 4h Donchian Channel (20-period) ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Upper band: highest high over 20 periods
    upper_band = pd.Series(high).rolling(window=20, min_periods=20).max().values
    # Lower band: lowest low over 20 periods
    lower_band = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === Volume confirmation (20-period average) ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === ATR (14-period) for stoploss ===
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(60, n):
        # Skip if indicators not ready
        if (np.isnan(upper_band[i]) or np.isnan(lower_band[i]) 
            or np.isnan(chop_1d_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        volume_now = volume[i]
        upper = upper_band[i]
        lower = lower_band[i]
        chop = chop_1d_aligned[i]
        vol_avg = vol_ma[i]
        
        # Regime filter: only trade in low-chop (trending) markets
        # CHOP < 38.2 = strong trend (trending regime)
        trending_regime = chop < 38.2
        
        # Volume confirmation: current volume > 2.0x average
        volume_confirmed = volume_now > 2.0 * vol_avg
        
        if position == 0:
            # Enter long on breakout above upper band in trending regime with volume
            long_condition = (price > upper) and trending_regime and volume_confirmed
            # Enter short on breakout below lower band in trending regime with volume
            short_condition = (price < lower) and trending_regime and volume_confirmed
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = price
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Check stoploss (2.0x ATR)
            if price < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trend exhaustion exit: price re-enters Donchian channel
            elif price < upper_band[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Check stoploss (2.0x ATR)
            if price > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trend exhaustion exit: price re-enters Donchian channel
            elif price > lower_band[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_VolumeSpike_RegimeFilter_V1"
timeframe = "4h"
leverage = 1.0