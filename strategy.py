#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h timeframe with 12h Choppiness Index regime filter + 4h Donchian breakout and volume confirmation.
# Uses 12h Choppiness Index to identify trending vs ranging markets: only trade breakouts when trending (CHOP < 38.2).
# In trending markets, Donchian(20) breakouts with volume confirmation capture momentum.
# In ranging markets (CHOP > 61.8), remain flat to avoid false breakouts.
# Designed to work in both bull and bear markets by adapting to market regime.
# Target: 75-200 total trades over 4 years (19-50/year) to minimize fee drag.
name = "4h_12h_Chop_Donchian20_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Choppiness Index calculation (called ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate True Range and ATR for Choppiness Index (period=14)
    tr1 = np.abs(high_12h - low_12h)
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr_12h = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_12h[0] = tr1_2h[0] if len(tr_12h) > 0 else tr1[0]  # Fix for first element
    
    atr_12h = pd.Series(tr_12h).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Choppiness Index: CHOP = 100 * log10(sum(ATR14) / (max(high) - min(low))) / log10(14)
    # We'll use a rolling window of 14 periods for the calculation
    atr_sum = pd.Series(atr_12h).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    range_12h = max_high - min_low
    range_12h = np.where(range_12h == 0, 1e-10, range_12h)
    
    chop_12h = 100 * np.log10(atr_sum / range_12h) / np.log10(14)
    
    # Align Choppiness Index to 4h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_12h, chop_12h)
    
    # Calculate Donchian channels on 4h (period=20)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.5 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(chop_aligned[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(volume_ma[i]):
            signals[i] = 0.0
            continue
            
        # Regime filter: only trade when market is trending (CHOP < 38.2)
        # Avoid ranging markets (CHOP > 61.8) where breakouts often fail
        trending_market = chop_aligned[i] < 38.2
        ranging_market = chop_aligned[i] > 61.8
        
        if position == 0:
            # Long when price breaks above Donchian high with volume confirmation in trending market
            if close[i] > donchian_high[i] and volume_confirm[i] and trending_market:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below Donchian low with volume confirmation in trending market
            elif close[i] < donchian_low[i] and volume_confirm[i] and trending_market:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit when price breaks below Donchian low or market becomes ranging
            if close[i] < donchian_low[i] or ranging_market:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit when price breaks above Donchian high or market becomes ranging
            if close[i] > donchian_high[i] or ranging_market:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals