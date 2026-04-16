#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d ATR-based volatility regime filter
# Uses 4h primary timeframe with 1d HTF for volatility regime confirmation.
# Donchian breakouts capture momentum in both bull and bear markets.
# ATR regime filter ensures we only trade when volatility is elevated (avoiding chop).
# ATR trailing stop (2.5x) protects gains and limits drawdown.
# Target: 75-200 total trades over 4 years (19-50/year) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # === 4h data (primary timeframe) ===
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # === 1d data (HTF for volatility regime) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # === 4h Donchian Channel (20-period) ===
    donch_upper = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donch_lower = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donch_upper_aligned = align_htf_to_ltf(prices, df_4h, donch_upper)
    donch_lower_aligned = align_htf_to_ltf(prices, df_4h, donch_lower)
    
    # === 1d ATR Regime Filter (High Volatility) ===
    # ATR(15) on 1d, then compare to its 50-period EMA
    atr_1d = np.abs(high_1d - low_1d)
    atr_ma_1d = pd.Series(atr_1d).ewm(span=15, adjust=False, min_periods=15).mean().values
    atr_ema_50 = pd.Series(atr_ma_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    vol_regime = atr_ma_1d > atr_ema_50  # High volatility regime
    vol_regime_aligned = align_htf_to_ltf(prices, df_1d, vol_regime.astype(float))
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 100
    
    # Track position and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0  # For long trailing stop
    lowest_since_entry = 0.0   # For short trailing stop
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(donch_upper_aligned[i]) or 
            np.isnan(donch_lower_aligned[i]) or
            np.isnan(vol_regime_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        vol_regime_val = vol_regime_aligned[i] > 0.5  # Boolean from aligned float
        
        # === ATR CALCULATION FOR TRAILING STOP ===
        atr_4h = np.abs(high_4h - low_4h)
        atr_ma = pd.Series(atr_4h).ewm(span=15, adjust=False, min_periods=15).mean().values
        atr_aligned = align_htf_to_ltf(prices, df_4h, atr_ma)
        atr_val = atr_aligned[i]
        
        # === TRAILING STOP LOGIC ===
        if position == 1:  # Long position
            # Update highest price since entry
            if price > highest_since_entry:
                highest_since_entry = price
            # Trail stop: exit if price drops 2.5*ATR from high
            if price < highest_since_entry - 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                continue
        
        elif position == -1:  # Short position
            # Update lowest price since entry
            if price < lowest_since_entry:
                lowest_since_entry = price
            # Trail stop: exit if price rises 2.5*ATR from low
            if price > lowest_since_entry + 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                lowest_since_entry = 0.0
                continue
        
        # === EXIT LOGIC (Donchian breakout in opposite direction) ===
        if position == 1:  # Long position
            # Exit when price breaks below Donchian lower
            if price < donch_lower_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                continue
        
        elif position == -1:  # Short position
            # Exit when price breaks above Donchian upper
            if price > donch_upper_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                lowest_since_entry = 0.0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Require high volatility regime for conviction
            if vol_regime_val:
                # Long on break above Donchian upper with high volatility
                if price > donch_upper_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                    highest_since_entry = price
                    continue
                # Short on break below Donchian lower with high volatility
                elif price < donch_lower_aligned[i]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = price
                    lowest_since_entry = price
                    continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_Donchian20_1dATRRegime_ATRTrail"
timeframe = "4h"
leverage = 1.0