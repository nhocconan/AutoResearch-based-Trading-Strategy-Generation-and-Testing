#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_Volume_ChopRegime_ATRStop
Hypothesis: 4h Donchian(20) breakouts with volume confirmation (>1.5x 20-period volume MA) and choppiness regime filter (CHOP > 61.8 for ranging markets = mean reversion at channel edges). 
In ranging markets (CHOP > 61.8), we fade the breakout: short at upper band, long at lower band. 
In trending markets (CHOP <= 61.8), we follow the breakout: long at upper band, short at lower band.
Uses 4h primary timeframe with 1d HTF for choppiness calculation (ATR-based). 
ATR-based stoploss: exit when price moves against position by 2.0 * ATR(14).
Target 20-50 trades/year (75-200 total over 4 years).
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
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    
    # ATR(14)
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Sum of ATR over 14 periods
    sum_atr_14 = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    
    # Max high - min low over 14 periods
    max_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    range_14 = max_high_14 - min_low_14
    
    # Choppiness Index: 100 * log10(sumATR14 / (maxHigh14 - minLow14)) / log10(14)
    chop = 100 * np.log10(sum_atr_14 / range_14) / np.log10(14)
    # Handle division by zero or invalid values
    chop = np.where((range_14 == 0) | np.isnan(chop) | np.isinf(chop), 50, chop)
    
    # Align choppiness to 4h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # === 4h Indicators (primary timeframe) ===
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Donchian Channel (20-period)
    donchian_upper = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Volume MA (20-period) for spike detection
    vol_ma = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for stoploss
    tr_4h1 = np.abs(high_4h - low_4h)
    tr_4h2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr_4h3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr_4h = np.maximum(tr_4h1, np.maximum(tr_4h2, tr_4h3))
    tr_4h[0] = tr_4h1[0]
    atr_4h = pd.Series(tr_4h).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if indicators not ready
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) 
            or np.isnan(vol_ma[i]) or np.isnan(atr_4h[i]) or np.isnan(chop_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_4h[i]
        vol = volume_4h[i]
        vol_ok = vol > 1.5 * vol_ma[i]  # volume spike confirmation
        
        if position == 0:
            # In ranging market (CHOP > 61.8): mean reversion at channel edges
            # Short at upper band, long at lower band
            if chop_aligned[i] > 61.8:
                if price >= donchian_upper[i] and vol_ok:
                    signals[i] = -0.25
                    position = -1
                    entry_price = price
                elif price <= donchian_lower[i] and vol_ok:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
            # In trending market (CHOP <= 61.8): follow breakout
            else:
                if price >= donchian_upper[i] and vol_ok:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                elif price <= donchian_lower[i] and vol_ok:
                    signals[i] = -0.25
                    position = -1
                    entry_price = price
        
        elif position == 1:
            # Long position: exit on stoploss or reverse signal
            # ATR-based stoploss: exit if price drops below entry - 2.0 * ATR
            if price <= entry_price - 2.0 * atr_4h[i]:
                signals[i] = 0.0
                position = 0
            # Reverse signal in ranging market: short at upper band
            elif chop_aligned[i] > 61.8 and price >= donchian_upper[i] and vol_ok:
                signals[i] = -0.25
                position = -1
                entry_price = price
            # Continue holding
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short position: exit on stoploss or reverse signal
            # ATR-based stoploss: exit if price rises above entry + 2.0 * ATR
            if price >= entry_price + 2.0 * atr_4h[i]:
                signals[i] = 0.0
                position = 0
            # Reverse signal in ranging market: long at lower band
            elif chop_aligned[i] > 61.8 and price <= donchian_lower[i] and vol_ok:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Continue holding
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_Volume_ChopRegime_ATRStop"
timeframe = "4h"
leverage = 1.0