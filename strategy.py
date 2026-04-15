#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian breakout (20) with 1w Supertrend(ATR=10, mult=3) trend filter and volume confirmation.
# Uses 1w Supertrend for robust trend bias in both bull/bear markets and Donchian breakouts for momentum entries.
# Volume filter (current volume > 1.3x 20-bar SMA) ensures breakouts have conviction.
# Designed for very low trade frequency (~15-25/year) to minimize fee drag. Works in bull/bear: Supertrend avoids counter-trend,
# Donchian captures breakouts with volume confirmation. Signal size 0.25 balances capture and drawdown control.

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
    if len(df_1d) < 50 or len(df_1w) < 50:
        return np.zeros(n)
    
    # === 1d Indicators: Donchian Channel (20) ===
    high_1d = pd.Series(df_1d['high'].values)
    low_1d = pd.Series(df_1d['low'].values)
    donchian_high = high_1d.rolling(window=20, min_periods=20).max().values
    donchian_low = low_1d.rolling(window=20, min_periods=20).min().values
    
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # === 1w Indicators: Supertrend(ATR=10, mult=3) ===
    # True Range
    tr1 = pd.Series(df_1w['high'].values) - pd.Series(df_1w['low'].values)
    tr2 = abs(pd.Series(df_1w['high'].values) - pd.Series(df_1w['close'].values).shift(1))
    tr3 = abs(pd.Series(df_1w['low'].values) - pd.Series(df_1w['close'].values).shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_10 = tr.rolling(window=10, min_periods=10).mean().values
    
    # Basic Upper and Lower Bands
    hl2 = (pd.Series(df_1w['high'].values) + pd.Series(df_1w['low'].values)) / 2
    upper_basic = hl2 + (3 * atr_10)
    lower_basic = hl2 - (3 * atr_10)
    
    # Initialize Supertrend
    supertrend = np.full_like(hl2.values, np.nan, dtype=float)
    direction = np.full_like(hl2.values, 1, dtype=int)  # 1 for uptrend, -1 for downtrend
    
    for i in range(1, len(hl2)):
        if np.isnan(atr_10[i-1]) or np.isnan(upper_basic[i-1]) or np.isnan(lower_basic[i-1]):
            continue
        # Upper Band
        if pd.Series(df_1w['close'].values).iloc[i-1] <= supertrend[i-1]:
            upper_basic[i] = min(upper_basic[i], upper_basic[i-1])
        else:
            upper_basic[i] = upper_basic[i]
        # Lower Band
        if pd.Series(df_1w['close'].values).iloc[i-1] >= supertrend[i-1]:
            lower_basic[i] = max(lower_basic[i], lower_basic[i-1])
        else:
            lower_basic[i] = lower_basic[i]
        # Supertrend
        if pd.Series(df_1w['close'].values).iloc[i] <= upper_basic[i]:
            supertrend[i] = upper_basic[i]
            direction[i] = -1
        else:
            supertrend[i] = lower_basic[i]
            direction[i] = 1
    
    # Align Supertrend and direction to 1d timeframe
    supertrend_aligned = align_htf_to_ltf(prices, df_1w, supertrend)
    direction_aligned = align_htf_to_ltf(prices, df_1w, direction)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    for i in range(warmup, n):
        # Volume filter: current volume > 1.3x 20-period volume SMA
        vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.3)
        
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(supertrend_aligned[i]) or np.isnan(direction_aligned[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above Donchian high (20-period breakout)
        # 2. 1w Supertrend indicates uptrend (direction = 1)
        # 3. Volume confirmation
        if (close[i] > donchian_high_aligned[i] and
            direction_aligned[i] == 1 and
            vol_confirm):
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below Donchian low (20-period breakdown)
        # 2. 1w Supertrend indicates downtrend (direction = -1)
        # 3. Volume confirmation
        elif (close[i] < donchian_low_aligned[i] and
              direction_aligned[i] == -1 and
              vol_confirm):
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "1d_Donchian20_Supertrend10_3_VolFilter_v1"
timeframe = "1d"
leverage = 1.0