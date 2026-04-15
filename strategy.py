#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R1/S1 breakout with volume confirmation and daily chop regime filter
# Uses proven pivot structure from HTF for institutional levels, volume to confirm breakout strength,
# and chop regime to avoid whipsaws in ranging markets. Designed for low trade frequency
# (target: 50-150 total over 4 years) to minimize fee drag while capturing breakout moves
# in both bull and bear regimes via bidirectional breakout logic.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels (R1, S1)
    # Camarilla: R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    hl_range = df_1d['high'] - df_1d['low']
    camarilla_r1 = df_1d['close'] + (1.1 * hl_range / 12)
    camarilla_s1 = df_1d['close'] - (1.1 * hl_range / 12)
    
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1.values)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1.values)
    
    # Calculate daily choppiness index regime filter
    # CHOP = 100 * log10(sum(ATR(14)) / log10(n) * (highest_high - lowest_low))
    # Simplified: CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    # We'll use trending regime (CHOP < 50) for breakout trading
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - np.concatenate([[df_1d['close'].iloc[0]], df_1d['close'].iloc[:-1]]))
    tr3 = np.abs(df_1d['low'] - np.concatenate([[df_1d['close'].iloc[0]], df_1d['close'].iloc[:-1]]))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean()
    
    # Sum of ATR over 14 periods
    sum_atr_14 = atr_14.rolling(window=14, min_periods=14).sum()
    highest_high_14 = df_1d['high'].rolling(window=14, min_periods=14).max()
    lowest_low_14 = df_1d['low'].rolling(window=14, min_periods=14).min()
    
    # Avoid division by zero
    chop_denom = np.log10(14) * (highest_high_14 - lowest_low_14)
    chop_denom = np.where(chop_denom == 0, 1e-10, chop_denom)
    chop_value = 100 * np.log10(sum_atr_14 / chop_denom)
    
    # Regime filter: trade only when market is trending (CHOP < 50)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_value.values.fillna(50))
    trending_regime = chop_aligned < 50
    
    # Calculate daily volume average for confirmation
    vol_ma_20 = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(vol_ma_20_aligned[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 1.5x daily average volume
        # Approximate: compare current volume to scaled daily average
        vol_filter = volume[i] > 1.5 * (vol_ma_20_aligned[i] / 8)  # 8x 12h bars in 1d
        
        # Regime filter: only trade in trending markets
        regime_filter = trending_regime[i]
        
        # Long conditions:
        # 1. Price breaks above Camarilla R1 level
        # 2. Volume confirmation
        # 3. Trending regime
        if (close[i] > camarilla_r1_aligned[i] and
            vol_filter and
            regime_filter):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. Price breaks below Camarilla S1 level
        # 2. Volume confirmation
        # 3. Trending regime
        elif (close[i] < camarilla_s1_aligned[i] and
              vol_filter and
              regime_filter):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "12h_Camarilla_R1S1_Volume_Regime_v1"
timeframe = "12h"
leverage = 1.0