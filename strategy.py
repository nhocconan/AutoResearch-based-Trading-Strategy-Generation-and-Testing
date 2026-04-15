#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1w Donchian channel (20) for trend direction and daily ATR(14) for volatility filtering.
# In 1w uptrend (price > weekly upper Donchian), go long when price breaks above daily Donchian(10) with volume confirmation.
# In 1w downtrend (price < weekly lower Donchian), go short when price breaks below daily Donchian(10) with volume confirmation.
# Uses discrete position sizing (0.25) to minimize fee drag. Designed for low trade frequency (10-20/year) to adapt to both bull and bear markets.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w and daily HTF data once before loop
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    if len(df_1w) < 30 or len(df_1d) < 30:
        return np.zeros(n)
    
    # === 1w Indicators: Donchian Channel (20) for trend ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    donchian_high_1w = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donchian_low_1w = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    donchian_high_1w_aligned = align_htf_to_ltf(prices, df_1w, donchian_high_1w)
    donchian_low_1w_aligned = align_htf_to_ltf(prices, df_1w, donchian_low_1w)
    
    # === Daily Indicators: Donchian Channel (10) for entry, ATR(14) for volatility filter ===
    donchian_high_1d = pd.Series(high).rolling(window=10, min_periods=10).max().values
    donchian_low_1d = pd.Series(low).rolling(window=10, min_periods=10).min().values
    
    # ATR(14) calculation
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 50
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_1w_aligned[i]) or np.isnan(donchian_low_1w_aligned[i]) or
            np.isnan(donchian_high_1d[i]) or np.isnan(donchian_low_1d[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: only trade when ATR > 0 (always true, but keeps structure)
        vol_filter = atr[i] > 0
        
        # Volume confirmation: current volume > 1.2x 20-period volume SMA
        vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.2)
        
        # === LONG CONDITIONS ===
        # 1. In 1w uptrend (price > weekly upper Donchian)
        # 2. Price breaks above daily Donchian(10)
        # 3. Volume confirmation
        if (close[i] > donchian_high_1w_aligned[i]) and (close[i] > donchian_high_1d[i]) and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. In 1w downtrend (price < weekly lower Donchian)
        # 2. Price breaks below daily Donchian(10)
        # 3. Volume confirmation
        elif (close[i] < donchian_low_1w_aligned[i]) and (close[i] < donchian_low_1d[i]) and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "1d_1wDonchian20_1dDonchian10_VolumeFilter_v1"
timeframe = "1d"
leverage = 1.0