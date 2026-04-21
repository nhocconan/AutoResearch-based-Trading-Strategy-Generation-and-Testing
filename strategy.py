#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_Volume_ChopRegime_ATRStop
Hypothesis: 4h Donchian(20) breakout with volume confirmation (>1.3x 20-period volume MA) and choppiness regime filter (CHOP > 61.8 for mean reversion, < 38.2 for trend following). 
In trending regimes (CHOP < 38.2): breakout in direction of trend (price > EMA50 for longs, < for shorts). 
In ranging regimes (CHOP > 61.8): mean reversion from Donchian extremes (short at upper band, long at lower band). 
ATR-based stoploss exits when price moves 2.5*ATR against position. 
Targets 20-50 trades/year (80-200 total over 4 years) by combining structure, volume, and regime filters.
Uses 4h primary timeframe with 1d HTF for EMA50 trend filter and choppiness calculation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for EMA50 and choppiness)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d EMA50 for trend filter ===
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === 1d Choppiness Index (CHOP) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar
    
    # ATR(14) sum
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: 100 * log10(atr_14 / (hh_14 - ll_14)) / log10(14)
    range_14 = hh_14 - ll_14
    chop = np.where(range_14 > 0, 100 * np.log10(atr_14 / range_14) / np.log10(14), 50)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # === 4h Indicators (primary timeframe) ===
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Donchian Channel (20)
    dc_upper = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    dc_lower = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Volume MA (20-period) for spike confirmation
    vol_ma = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for stoploss
    tr_4h1 = np.abs(high_4h - low_4h)
    tr_4h2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr_4h3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr_4h = np.maximum(tr_4h1, np.maximum(tr_4h2, tr_4h3))
    tr_4h[0] = tr_4h1[0]
    atr_14_4h = pd.Series(tr_4h).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(dc_upper[i]) or np.isnan(dc_lower[i]) or np.isnan(vol_ma[i]) 
            or np.isnan(ema_50_1d_aligned[i]) or np.isnan(chop_aligned[i]) 
            or np.isnan(atr_14_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_4h[i]
        vol = volume_4h[i]
        vol_ok = vol > 1.3 * vol_ma[i]  # volume spike confirmation
        
        if position == 0:
            # Determine regime
            if chop_aligned[i] < 38.2:  # Trending regime
                # Long: break above upper Donchian + volume + uptrend (price > EMA50)
                if price > dc_upper[i] and vol_ok and price > ema_50_1d_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                # Short: break below lower Donchian + volume + downtrend (price < EMA50)
                elif price < dc_lower[i] and vol_ok and price < ema_50_1d_aligned[i]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = price
            elif chop_aligned[i] > 61.8:  # Ranging regime
                # Long: mean reversion from lower Donchian (oversold)
                if price < dc_lower[i] and vol_ok:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                # Short: mean reversion from upper Donchian (overbought)
                elif price > dc_upper[i] and vol_ok:
                    signals[i] = -0.25
                    position = -1
                    entry_price = price
        
        elif position == 1:
            # Exit conditions: stoploss or mean reversion signal
            stop_loss = entry_price - 2.5 * atr_14_4h[i]
            if price < stop_loss or (chop_aligned[i] > 61.8 and price > dc_upper[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit conditions: stoploss or mean reversion signal
            stop_loss = entry_price + 2.5 * atr_14_4h[i]
            if price > stop_loss or (chop_aligned[i] > 61.8 and price < dc_lower[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_Volume_ChopRegime_ATRStop"
timeframe = "4h"
leverage = 1.0