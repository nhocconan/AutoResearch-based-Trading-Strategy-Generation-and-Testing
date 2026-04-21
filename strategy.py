#!/usr/bin/env python3
"""
6h_Donchian20_WeeklyPivot_Confluence_v1
Hypothesis: On 6h timeframe, price breaking above Donchian(20) high or below Donchian(20) low captures medium-term momentum. Combined with weekly Camarilla pivot (R4/S4) for institutional level confirmation and volume spike filter (>1.8x 20-period average) to avoid false breakouts. Works in both bull (breakout continuation) and bear (breakdown continuation) regimes by following price structure. Designed for low trade frequency (~15-25/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for Donchian, 1w for weekly pivot)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # === Donchian(20) from 1d timeframe ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Donchian high: 20-period rolling maximum of high
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    # Donchian low: 20-period rolling minimum of low
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # === Weekly Camarilla pivot levels (R4/S4) from 1w timeframe ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly Camarilla R4 and S4 levels
    weekly_range = high_1w - low_1w
    camarilla_r4 = close_1w + weekly_range * 1.1 / 2  # R4 = Close + 1.1*(Range)/2
    camarilla_s4 = close_1w - weekly_range * 1.1 / 2  # S4 = Close - 1.1*(Range)/2
    
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s4)
    
    # === Volume spike filter (20-period) ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.where(vol_ma > 0, volume / vol_ma, 0)
    
    # === ATR for stoploss (14-period) ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        price_high = prices['high'].iloc[i]
        price_low = prices['low'].iloc[i]
        donchian_high_val = donchian_high_aligned[i]
        donchian_low_val = donchian_low_aligned[i]
        r4 = camarilla_r4_aligned[i]
        s4 = camarilla_s4_aligned[i]
        vol_spike = vol_ratio[i]
        atr_val = atr[i]
        
        if position == 0:
            # Long: price breaks above Donchian high AND above weekly R4 + volume spike > 1.8
            if price_close > donchian_high_val and price_close > r4 and vol_spike > 1.8:
                signals[i] = 0.25
                position = 1
                entry_price = price_close
            # Short: price breaks below Donchian low AND below weekly S4 + volume spike > 1.8
            elif price_close < donchian_low_val and price_close < s4 and vol_spike > 1.8:
                signals[i] = -0.25
                position = -1
                entry_price = price_close
        
        elif position != 0:
            # Stoploss: 2.5 * ATR from entry
            if position == 1:
                if price_close < entry_price - 2.5 * atr_val:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if price_close > entry_price + 2.5 * atr_val:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_WeeklyPivot_Confluence_v1"
timeframe = "6h"
leverage = 1.0