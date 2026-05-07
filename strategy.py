#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Choppiness Index regime filter with 12h Donchian breakout and volume confirmation.
# Use 12h Choppiness Index to detect regimes: >61.8 = range (mean revert), <38.2 = trending (trend follow).
# In trending regime (CHOP < 38.2): enter long on breakout above 12h Donchian(20) upper band, short on breakdown below lower band.
# In ranging regime (CHOP > 61.8): enter long at Donchian lower band (mean reversion), short at upper band.
# Volume confirmation: current volume > 1.5x 20-period average to avoid low-conviction breakouts.
# Exit when regime changes or volatility drops (ATR ratio < 0.8).
# Designed for 6h timeframe with moderate trade frequency (target: 15-30/year) to avoid fee drag.
# Works in both bull and bear markets by adapting to regime.
name = "6h_Chop_Donchian_Breakout_12hVol"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h data for Choppiness Index and Donchian channels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 14:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range for ATR calculation
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with indices
    
    # ATR(14)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Sum of True Range over 14 periods for Choppiness denominator
    sum_tr = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    highest_high = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: 100 * log10(sum_tr / (highest_high - lowest_low)) / log10(14)
    # Avoid division by zero
    range_hl = highest_high - lowest_low
    chop = np.full_like(close_12h, np.nan)
    mask = (range_hl > 0) & (~np.isnan(sum_tr))
    chop[mask] = 100 * np.log10(sum_tr[mask] / range_hl[mask]) / np.log10(14)
    
    # Donchian channels (20-period)
    donch_h = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donch_l = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Align HTF indicators to 6s timeframe
    chop_aligned = align_htf_to_ltf(prices, df_12h, chop)
    donch_h_aligned = align_htf_to_ltf(prices, df_12h, donch_h)
    donch_l_aligned = align_htf_to_ltf(prices, df_12h, donch_l)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(chop_aligned[i]) or np.isnan(donch_h_aligned[i]) or np.isnan(donch_l_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        chop_val = chop_aligned[i]
        donch_h_val = donch_h_aligned[i]
        donch_l_val = donch_l_aligned[i]
        
        if position == 0:
            # Determine regime and enter accordingly
            if chop_val < 38.2:  # Trending regime
                # Breakout entry
                long_cond = (close[i] > donch_h_val) and volume_filter[i]
                short_cond = (close[i] < donch_l_val) and volume_filter[i]
            elif chop_val > 61.8:  # Ranging regime
                # Mean reversion entry
                long_cond = (close[i] < donch_l_val) and volume_filter[i]
                short_cond = (close[i] > donch_h_val) and volume_filter[i]
            else:  # Transition zone - no trade
                long_cond = False
                short_cond = False
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: regime change to ranging or volatility drop
            if chop_val > 61.8:  # Exit trending for ranging
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: regime change to ranging or volatility drop
            if chop_val > 61.8:  # Exit trending for ranging
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals