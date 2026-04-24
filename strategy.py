#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d ATR-based volume confirmation and choppiness regime filter.
- Primary timeframe: 12h for lower trade frequency and reduced fee drag.
- HTF: 1d for ATR(14) volume spike (>2.0x 24-period MA) and choppiness index (14) regime.
- Entry: Long when price breaks above Donchian(20) high AND volume spike AND chop < 61.8 (trending).
         Short when price breaks below Donchian(20) low AND volume spike AND chop < 61.8 (trending).
- Exit: Opposite Donchian breakout (long exits on lower band, short on upper band) or loss of volume confirmation.
- Signal size: 0.25 discrete to balance return and drawdown.
- No session filter to maximize opportunity in 12h timeframe.
- Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.
This strategy captures strong trending moves with volume confirmation while avoiding choppy regimes,
working in both bull and bear markets by only taking breakout trades with institutional participation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 12h Donchian channels (20-period)
    donchian_h = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_l = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Get 1d data for ATR and choppiness
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ATR(14)
    df_1d_high = df_1d['high'].values
    df_1d_low = df_1d['low'].values
    df_1d_close = df_1d['close'].values
    tr1 = np.abs(df_1d_high - df_1d_low)
    tr2 = np.abs(df_1d_high - np.roll(df_1d_close, 1))
    tr3 = np.abs(df_1d_low - np.roll(df_1d_close, 1))
    tr1[0] = tr2[0] = tr3[0] = 0.0  # first bar
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 1d choppiness index (14)
    sum_tr = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    hh = pd.Series(df_1d_high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(df_1d_low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(sum_tr / (hh - ll)) / np.log10(14)
    # Handle division by zero or invalid values
    chop = np.where((hh - ll) > 0, chop, 50.0)
    
    # Calculate 24-period 1d volume MA
    df_1d_volume = df_1d['volume'].values
    vol_ma_1d = pd.Series(df_1d_volume).rolling(window=24, min_periods=24).mean().values
    
    # Align HTF indicators to 12h
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Volume confirmation: current 12h volume > 2.0 * 24-period 1d volume MA (aligned)
    volume_spike = volume > (2.0 * vol_ma_1d_aligned)
    
    # Regime filter: choppiness < 61.8 indicates trending market
    trending_regime = chop_aligned < 61.8
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 14, 24)  # Need enough bars for Donchian, ATR/chop, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_h[i]) or np.isnan(donchian_l[i]) or 
            np.isnan(atr_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        if position == 0:
            # Check for entry signals with volume spike and trending regime
            if volume_spike[i] and trending_regime[i]:
                # Bullish breakout: price breaks above Donchian high
                if curr_high > donchian_h[i]:
                    signals[i] = 0.25
                    position = 1
                # Bearish breakout: price breaks below Donchian low
                elif curr_low < donchian_l[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price breaks below Donchian low OR loss of volume confirmation OR chop becomes too high
            if curr_low < donchian_l[i] or not volume_spike[i] or not trending_regime[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above Donchian high OR loss of volume confirmation OR chop becomes too high
            if curr_high > donchian_h[i] or not volume_spike[i] or not trending_regime[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_1dATR_VolumeSpike_ChopRegime_v1"
timeframe = "12h"
leverage = 1.0