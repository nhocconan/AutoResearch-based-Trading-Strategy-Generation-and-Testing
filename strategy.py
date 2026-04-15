#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with volume confirmation and ATR-based volatility filter
# Long when price breaks above 12h Donchian(20) high + volume > 1.3x 20-period volume SMA + ATR(14) > 0.5 * ATR(50)
# Short when price breaks below 12h Donchian(20) low + volume > 1.3x 20-period volume SMA + ATR(14) > 0.5 * ATR(50)
# Uses 1d ATR for volatility regime detection to avoid look-ahead
# Designed for low trade frequency (12-37/year) to minimize fee drag while capturing breakouts in volatile regimes
# Works in both bull and bear markets by trading volatility expansion breakouts

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h and 1d HTF data once before loop
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_12h) < 30 or len(df_1d) < 30:
        return np.zeros(n)
    
    # === 12h Indicators: Donchian Channel (20) ===
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    donchian_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    
    # === 1d Indicators: ATR for Volatility Regime ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = high_1d[0] - low_1d[0]  # first TR
    
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_50 = pd.Series(tr).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volatility filter: short-term ATR > 50% of long-term ATR (expanding volatility)
    vol_expansion = atr_14 > (0.5 * atr_50)
    vol_expansion_aligned = align_htf_to_ltf(prices, df_1d, vol_expansion)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    for i in range(warmup, n):
        # Volume filter: current volume > 1.3x 20-period volume SMA
        vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.3)
        
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(vol_expansion_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above 12h upper Donchian
        # 2. Volume confirmation
        # 3. Volatility expansion regime
        if (close[i] > donchian_high_aligned[i]) and vol_confirm and vol_expansion_aligned[i]:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below 12h lower Donchian
        # 2. Volume confirmation
        # 3. Volatility expansion regime
        elif (close[i] < donchian_low_aligned[i]) and vol_confirm and vol_expansion_aligned[i]:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "12h_Donchian20_Volume_VolatilityFilter_v1"
timeframe = "12h"
leverage = 1.0