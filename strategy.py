#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Choppiness Index regime filter + Donchian(20) breakout + volume confirmation
# Choppiness Index identifies ranging (high CHOP) vs trending (low CHOP) markets.
# In trending regimes (CHOP < 38.2), we trade Donchian breakouts with volume filter.
# In ranging regimes (CHOP > 61.8), we fade Donchian breakouts (mean reversion).
# This adapts to both bull and bear markets by using regime-appropriate logic.
# Target: 20-50 trades/year on 1d timeframe to minimize fee drag.
name = "1d_ChopRegime_Donchian20_VolumeFilter"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for regime filter (higher timeframe)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate Choppiness Index on weekly data
    # CHOP = 100 * log10(SUM(TR(14)) / (HHV(HIGH,14) - LLV(LOW,14))) / log10(14)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # ATR-like sum of TR over 14 periods
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index
    chop = 100 * np.log10(tr_sum / (hh - ll)) / np.log10(14)
    chop = np.where((hh - ll) == 0, 50, chop)  # Avoid division by zero
    chop = np.where(np.isnan(chop), 50, chop)   # Default to neutral
    
    # Align weekly chop to daily timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop)
    
    # Donchian channels on daily data (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 2.0 x 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure sufficient warmup for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(chop_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        chop_val = chop_aligned[i]
        upper = donchian_high[i]
        lower = donchian_low[i]
        
        volume_confirmed = vol > 2.0 * vol_ma
        
        if position == 0:
            # Trending regime (CHOP < 38.2): trade breakouts
            if chop_val < 38.2:
                # Long breakout
                if price > upper and volume_confirmed:
                    signals[i] = 0.25
                    position = 1
                # Short breakout
                elif price < lower and volume_confirmed:
                    signals[i] = -0.25
                    position = -1
            # Ranging regime (CHOP > 61.8): fade breakouts (mean reversion)
            elif chop_val > 61.8:
                # Long at lower band (oversold)
                if price < lower and volume_confirmed:
                    signals[i] = 0.20
                    position = 1
                # Short at upper band (overbought)
                elif price > upper and volume_confirmed:
                    signals[i] = -0.20
                    position = -1
        
        elif position == 1:
            # Exit conditions
            if chop_val < 38.2:
                # Trending: exit when price crosses midpoint
                midpoint = (upper + lower) / 2
                if price < midpoint:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:
                # Ranging: exit when price returns to opposite band
                if price > upper:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
        
        elif position == -1:
            # Exit conditions
            if chop_val < 38.2:
                # Trending: exit when price crosses midpoint
                midpoint = (upper + lower) / 2
                if price > midpoint:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:
                # Ranging: exit when price returns to opposite band
                if price < lower:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals