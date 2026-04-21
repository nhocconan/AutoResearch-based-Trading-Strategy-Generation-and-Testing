#!/usr/bin/env python3
"""
6h_ADX_Regime_Donchian20_Breakout_VolumeFilter_v1
Hypothesis: 6h Donchian(20) breakouts filtered by 12h ADX regime (ADX>25 = trending, ADX<20 = range) and volume spike (>1.5x average).
In trending regime (ADX>25): breakout continuation logic (price > upper band for long, < lower band for short).
In range regime (ADX<20): mean reversion at bands (price < upper band for short, > lower band for long).
Uses discrete position sizing (0.0, ±0.25) to minimize fee churn and overtrading.
ATR-based trailing stop with 2.0x ATR distance. Designed for 50-150 total trades over 4 years.
Works in bull/bear via regime adaptation and volume confirmation to avoid false breakouts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (12h for ADX regime)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # === 12h ADX for regime filter ===
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = pd.Series(high_12h - low_12h)
    tr2 = pd.Series(np.abs(high_12h - np.roll(close_12h, 1)))
    tr3 = pd.Series(np.abs(low_12h - np.roll(close_12h, 1)))
    tr_12h = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_12h = tr_12h.rolling(window=14, min_periods=14).mean().values
    
    # Directional Movement
    dm_plus = pd.Series(np.where((high_12h - np.roll(high_12h, 1)) > (np.roll(low_12h, 1) - low_12h),
                                 np.maximum(high_12h - np.roll(high_12h, 1), 0), 0))
    dm_minus = pd.Series(np.where((np.roll(low_12h, 1) - low_12h) > (high_12h - np.roll(high_12h, 1)),
                                  np.maximum(np.roll(low_12h, 1) - low_12h, 0), 0))
    
    # Smoothed DM and TR
    dm_plus_smooth = dm_plus.ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_minus_smooth = dm_minus.ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    atr_smooth = tr_12h.ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / atr_smooth
    di_minus = 100 * dm_minus_smooth / atr_smooth
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align ADX to 6h timeframe (use previous completed 12h bar)
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    # === 6h Donchian Channel (20-period) ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === ATR (14-period) for stoploss ===
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # === Volume spike filter ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(adx_aligned[i]) or np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i])
            or np.isnan(atr[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_spike = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Regime-based entry logic
            if adx_aligned[i] > 25:  # Trending regime
                # Breakout continuation
                long_breakout = price > donchian_upper[i]
                short_breakout = price < donchian_lower[i]
                
                if long_breakout and vol_spike:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                elif short_breakout and vol_spike:
                    signals[i] = -0.25
                    position = -1
                    entry_price = price
            elif adx_aligned[i] < 20:  # Range regime
                # Mean reversion at bands
                long_reversion = price < donchian_lower[i]
                short_reversion = price > donchian_upper[i]
                
                if long_reversion and vol_spike:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                elif short_reversion and vol_spike:
                    signals[i] = -0.25
                    position = -1
                    entry_price = price
        
        elif position == 1:
            # Check stoploss
            if price < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trailing exit: price closes below Donchian lower (support broken)
            elif price < donchian_lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Check stoploss
            if price > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trailing exit: price closes above Donchian upper (resistance broken)
            elif price > donchian_upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ADX_Regime_Donchian20_Breakout_VolumeFilter_v1"
timeframe = "6h"
leverage = 1.0