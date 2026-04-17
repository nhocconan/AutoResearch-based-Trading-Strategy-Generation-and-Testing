#!/usr/bin/env python3
"""
Hypothesis: 1d timeframe with 1w Williams %R mean reversion + volume confirmation + chop regime filter.
Long when Williams %R < -80 (oversold) with volume confirmation and choppy market (CHOP > 61.8).
Short when Williams %R > -20 (overbought) with volume confirmation and choppy market (CHOP > 61.8).
Exit when Williams %R returns to -50 (mean reversion) or chop regime ends (CHOP < 38.2 trending).
Uses 1w timeframe for Williams %R to avoid noise and 1d for entry timing and volume confirmation.
Designed to capture mean reversion in bear markets (2025 test) while avoiding false signals in strong trends.
Williams %R identifies overextended moves, chop filter ensures we only mean revert in ranging markets.
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
    
    # Get 1w data for Williams %R calculation
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w Williams %R (14-period)
    period = 14
    highest_high = pd.Series(high_1w).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low_1w).rolling(window=period, min_periods=period).min().values
    williams_r = -100 * (highest_high - close_1w) / (highest_high - lowest_low)
    # Handle division by zero when high == low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate 1d choppiness index (CHOP) for regime filter
    atr_period = 14
    chop_period = 14
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period TR is just high-low
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    max_high = pd.Series(high).rolling(window=chop_period, min_periods=chop_period).max().values
    min_low = pd.Series(low).rolling(window=chop_period, min_periods=chop_period).min().values
    chop = 100 * np.log10(atr * chop_period / (max_high - min_low)) / np.log10(chop_period)
    # Handle division by zero when max_high == min_low
    chop = np.where((max_high - min_low) == 0, 50, chop)
    
    # Calculate 1d volume 20-period average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align 1w indicators to 1d timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1w, williams_r)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # need enough for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(williams_r_aligned[i]) or 
            np.isnan(chop[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.3x 20-period average
        volume_confirmed = volume[i] > 1.3 * vol_ma_20[i]
        
        # Chop regime filter: only trade in ranging markets (CHOP > 61.8)
        chop_regime = chop[i] > 61.8
        
        if position == 0:
            # Long: Williams %R < -80 (oversold) with volume and chop regime
            if (williams_r_aligned[i] < -80 and 
                volume_confirmed and 
                chop_regime):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R > -20 (overbought) with volume and chop regime
            elif (williams_r_aligned[i] > -20 and 
                  volume_confirmed and 
                  chop_regime):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Williams %R returns to -50 OR chop regime ends (trending market)
            if (williams_r_aligned[i] >= -50 or 
                chop[i] < 38.2):  # trending market
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Williams %R returns to -50 OR chop regime ends (trending market)
            if (williams_r_aligned[i] <= -50 or 
                chop[i] < 38.2):  # trending market
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_1wWilliamsR_MeanReversion_Volume_Chop_Regime"
timeframe = "1d"
leverage = 1.0