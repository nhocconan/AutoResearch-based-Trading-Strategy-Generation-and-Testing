#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with volume confirmation and 1w Supertrend filter
# Long when price breaks above 20-bar high + volume > 1.5x 20-period avg + Supertrend(1w) bullish
# Short when price breaks below 20-bar low + volume > 1.5x 20-period avg + Supertrend(1w) bearish
# Uses volume to confirm institutional interest and Supertrend for trend alignment
# Designed for low trade frequency (15-25/year) with clear entry/exit rules to minimize fee drag
# Works in bull markets via breakouts and in bear markets via short breakdowns with trend filter

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d and 1w HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 30 or len(df_1w) < 30:
        return np.zeros(n)
    
    # === 1d Indicators: Donchian Channel (20-period) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Donchian calculations (20-period high/low)
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # === 1w Indicators: Supertrend (10, 3.0) ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate ATR for Supertrend
    tr1 = pd.Series(high_1w - low_1w)
    tr2 = pd.Series(np.abs(high_1w - pd.Series(close_1w).shift(1)))
    tr3 = pd.Series(np.abs(low_1w - pd.Series(close_1w).shift(1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1w = tr.ewm(alpha=1/10, adjust=False, min_periods=10).mean().values
    
    # Supertrend calculation
    hl2 = (high_1w + low_1w) / 2
    upper_band = hl2 + (3.0 * atr_1w)
    lower_band = hl2 - (3.0 * atr_1w)
    
    # Initialize Supertrend arrays
    supertrend = np.zeros_like(close_1w)
    direction = np.ones_like(close_1w)  # 1 for uptrend, -1 for downtrend
    
    # Set first value
    supertrend[0] = hl2[0]
    direction[0] = 1
    
    for i in range(1, len(close_1w)):
        if close_1w[i] > supertrend[i-1]:
            direction[i] = 1
        else:
            direction[i] = -1
        
        if direction[i] == 1:
            supertrend[i] = max(lower_band[i], supertrend[i-1])
        else:
            supertrend[i] = min(upper_band[i], supertrend[i-1])
    
    supertrend_aligned = align_htf_to_ltf(prices, df_1w, supertrend)
    supertrend_direction_aligned = align_htf_to_ltf(prices, df_1w, direction)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    for i in range(warmup, n):
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(supertrend_aligned[i]) or np.isnan(supertrend_direction_aligned[i]) or
            np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above 1d Donchian high (20-period)
        # 2. Volume confirmation
        # 3. 1w Supertrend is bullish (direction = 1)
        if (close[i] > donchian_high_aligned[i]) and vol_confirm and (supertrend_direction_aligned[i] == 1):
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below 1d Donchian low (20-period)
        # 2. Volume confirmation
        # 3. 1w Supertrend is bearish (direction = -1)
        elif (close[i] < donchian_low_aligned[i]) and vol_confirm and (supertrend_direction_aligned[i] == -1):
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "12h_Donchian20_Volume_1wSupertrend_Filter_v1"
timeframe = "12h"
leverage = 1.0