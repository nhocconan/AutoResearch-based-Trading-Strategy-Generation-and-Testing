#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_VolumeSpike_ChopRegime
Hypothesis: On 4h timeframe, Donchian(20) breakouts with volume confirmation (>2.0x 20-bar average volume) and choppiness regime filter (CHOP > 61.8 for mean reversion) captures breakouts with institutional participation while avoiding whipsaws in strong trends. Designed for 20-40 trades/year to minimize fee drag. Works in both bull and bear markets via regime-adaptive logic: in ranging markets (CHOP > 61.8), we trade mean reversion at Donchian channels; in trending markets (CHOP < 38.2), we trade breakouts with the trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for HTF trend and choppiness calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate EMA50 on 1d for HTF trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate ATR(14) for Donchian channel width and choppiness
    tr1 = pd.Series(high_1d - low_1d)
    tr2 = pd.Series(abs(high_1d - close_1d.shift(1)))
    tr3 = pd.Series(abs(low_1d - close_1d.shift(1)))
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # Calculate choppiness index: CHOP = 100 * log10(sum(ATR14) / (max(high)-min(low))) / log10(14)
    max_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    sum_atr_14 = pd.Series(atr_14_1d).rolling(window=14, min_periods=14).sum().values
    chop_1d = 100 * np.log10(sum_atr_14 / (max_high_14 - min_low_14)) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Calculate Donchian(20) channels from 1d data
    # Upper = max(high, 20), Lower = min(low, 20)
    donch_high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donch_low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high_20)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low_20)
    
    # Volume average (20-period) for volume spike filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for calculations
    start_idx = max(50, 20)  # EMA50, ATR14, Donchian20, vol MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_aligned[i]) or 
            np.isnan(chop_aligned[i]) or 
            np.isnan(donch_high_aligned[i]) or 
            np.isnan(donch_low_aligned[i]) or 
            np.isnan(vol_ma[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Get aligned values
        ema_val = ema_50_aligned[i]
        chop_val = chop_aligned[i]
        donch_high_val = donch_high_aligned[i]
        donch_low_val = donch_low_aligned[i]
        vol_ma_val = vol_ma[i]
        vol_val = volume[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        
        # Volume spike condition: current volume > 2.0x 20-period average
        volume_spike = vol_val > 2.0 * vol_ma_val
        
        # Regime conditions
        is_ranging = chop_val > 61.8  # Mean reversion regime
        is_trending = chop_val < 38.2  # Trending regime
        
        if position == 0:
            # Look for entry signals based on regime
            if is_ranging:
                # In ranging markets: mean reversion at Donchian channels
                # Long: price touches lower band with volume spike
                long_signal = (low_val <= donch_low_val) and volume_spike
                # Short: price touches upper band with volume spike
                short_signal = (high_val >= donch_high_val) and volume_spike
            elif is_trending:
                # In trending markets: breakouts with trend
                # Long: price breaks above upper band with uptrend (close > EMA50) and volume spike
                long_signal = (high_val > donch_high_val) and (close_val > ema_val) and volume_spike
                # Short: price breaks below lower band with downtrend (close < EMA50) and volume spike
                short_signal = (low_val < donch_low_val) and (close_val < ema_val) and volume_spike
            else:
                # In transition regime (38.2 <= CHOP <= 61.8): no trading
                long_signal = False
                short_signal = False
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit conditions:
            # 1. Opposite Donchian touch/break: price touches or breaks lower band
            if low_val <= donch_low_val:
                signals[i] = 0.0
                position = 0
            # 2. Trend failure: price closes below EMA50 in trending market
            elif is_trending and close_val < ema_val:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions:
            # 1. Opposite Donchian touch/break: price touches or breaks upper band
            if high_val >= donch_high_val:
                signals[i] = 0.0
                position = 0
            # 2. Trend failure: price closes above EMA50 in trending market
            elif is_trending and close_val > ema_val:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Donchian20_Breakout_VolumeSpike_ChopRegime"
timeframe = "4h"
leverage = 1.0