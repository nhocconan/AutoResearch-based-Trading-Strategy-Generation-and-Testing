#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + volume confirmation + choppiness regime filter
# Long when price breaks above Donchian(20) high AND volume > 1.5x 20-period average AND CHOP(14) > 61.8 (range regime)
# Short when price breaks below Donchian(20) low AND volume > 1.5x 20-period average AND CHOP(14) > 61.8 (range regime)
# Exit when price crosses Donchian(20) midpoint OR choppiness regime shifts to trending (CHOP < 38.2)
# Uses 4h primary timeframe with 1d HTF for choppiness calculation to capture mean-reversion in ranging markets
# Discrete sizing (0.25) to limit fee drag and manage drawdown in both bull and bear markets
# Target: 75-200 total trades over 4 years (19-50/year) to avoid fee drag
# Donchian channels provide clear breakout levels; volume confirms participation; chop filter ensures mean-reversion edge

name = "4h_Donchian20_Volume_Chop_MeanReversion"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for choppiness calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate Choppiness Index on 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # align with 1d bars
    
    # ATR(14)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Highest high and lowest low over 14 periods
    hh_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: 100 * log10(sum(TR(14)) / (HH(14) - LL(14))) / log10(14)
    sum_tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    chop_denom = hh_14 - ll_14
    chop_denom = np.where(chop_denom == 0, 1e-10, chop_denom)  # avoid division by zero
    chop = 100 * np.log10(sum_tr_14 / chop_denom) / np.log10(14)
    
    # Align chop to 4h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Donchian(20) on 4h data
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_high + donch_low) / 2.0
    
    # Volume confirmation: volume > 1.5x 20-period average
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or np.isnan(donch_mid[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Regime filter: only trade in ranging markets (CHOP > 61.8)
        in_range_regime = chop_aligned[i] > 61.8
        
        if position == 0:
            # Long conditions: break above Donchian high AND volume spike AND range regime
            if close[i] > donch_high[i] and volume_filter[i] and in_range_regime:
                signals[i] = 0.25
                position = 1
            # Short conditions: break below Donchian low AND volume spike AND range regime
            elif close[i] < donch_low[i] and volume_filter[i] and in_range_regime:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses Donchian midpoint OR regime shifts to trending (CHOP < 38.2)
            if close[i] < donch_mid[i] or chop_aligned[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses Donchian midpoint OR regime shifts to trending (CHOP < 38.2)
            if close[i] > donch_mid[i] or chop_aligned[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals