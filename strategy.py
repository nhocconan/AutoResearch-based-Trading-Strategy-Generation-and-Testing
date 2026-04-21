#!/usr/bin/env python3
"""
1d_KAMA_Regime_Filter_DonchianExit
Hypothesis: On daily timeframe, use KAMA trend direction + choppiness regime filter for entry, 
and Donchian(20) breakout in opposite direction for exit. Works in both bull and bear by 
adapting to market regime (choppy = mean revert, trending = follow KAMA). 
Volume confirmation ensures institutional participation. Designed for low trade frequency 
(15-25/year) to minimize fee drag on BTC/ETH.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for KAMA, chop, Donchian)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d OHLC for indicators ===
    df_1d_close = df_1d['close'].values
    df_1d_high = df_1d['high'].values
    df_1d_low = df_1d['low'].values
    df_1d_volume = df_1d['volume'].values
    
    # === KAMA (adaptive trend) ===
    close_s = pd.Series(df_1d_close)
    change = np.abs(close_s.diff(10).values)
    volatility = np.abs(close_s.diff(1)).rolling(window=10, min_periods=1).sum().values
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.zeros_like(df_1d_close)
    kama[0] = df_1d_close[0]
    for i in range(1, len(df_1d_close)):
        kama[i] = kama[i-1] + sc[i] * (df_1d_close[i] - kama[i-1])
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # === Choppiness Index (regime filter) ===
    atr_14 = pd.Series(np.maximum.reduce([
        df_1d_high - df_1d_low,
        np.abs(df_1d_high - np.roll(df_1d_close, 1)),
        np.abs(df_1d_low - np.roll(df_1d_close, 1))
    ]).rolling(window=14, min_periods=14).sum())
    max_hh = pd.Series(df_1d_high).rolling(window=14, min_periods=14).max()
    min_ll = pd.Series(df_1d_low).rolling(window=14, min_periods=14).min()
    chop = 100 * np.log10(atr_14 / (max_hh - min_ll)) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop.values)
    
    # === Donchian(20) for exit ===
    donch_high = pd.Series(df_1d_high).rolling(window=20, min_periods=20).max()
    donch_low = pd.Series(df_1d_low).rolling(window=20, min_periods=20).min()
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high.values)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low.values)
    
    # === Volume confirmation (1.5x 20-period MA) ===
    vol_ma = pd.Series(df_1d_volume).rolling(window=20, min_periods=20).mean()
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(vol_ma_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = df_1d_close[i]  # Use 1d close for signal generation (aligned to 1d)
        volume_now = df_1d_volume[i]
        kama_val = kama_aligned[i]
        chop_val = chop_aligned[i]
        donch_high_val = donch_high_aligned[i]
        donch_low_val = donch_low_aligned[i]
        vol_avg = vol_ma_aligned[i]
        
        # Volume confirmation
        volume_ok = volume_now > 1.5 * vol_avg
        
        # Regime: chop > 61.8 = ranging (mean revert), chop < 38.2 = trending
        is_ranging = chop_val > 61.8
        is_trending = chop_val < 38.2
        
        if position == 0:
            # Enter based on regime
            if is_ranging:
                # Mean reversion: price deviated from KAMA
                long_condition = (price < kama_val * 0.98) and volume_ok
                short_condition = (price > kama_val * 1.02) and volume_ok
            elif is_trending:
                # Follow trend: price > KAMA for long, price < KAMA for short
                long_condition = (price > kama_val) and volume_ok
                short_condition = (price < kama_val) and volume_ok
            else:
                # Transition regime - wait for clearer signal
                long_condition = False
                short_condition = False
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = price
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Exit: Donchian breakout in opposite direction OR regime shift to extreme ranging
            if price < donch_low_val or chop_val > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Donchian breakout in opposite direction OR regime shift to extreme ranging
            if price > donch_high_val or chop_val > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_KAMA_Regime_Filter_DonchianExit"
timeframe = "1d"
leverage = 1.0