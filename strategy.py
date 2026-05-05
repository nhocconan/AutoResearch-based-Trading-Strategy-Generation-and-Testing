#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d ATR(14) volatility filter and volume confirmation
# Long when price breaks above 4h Donchian upper channel AND 1d ATR ratio > 1.2 AND volume > 1.3x 20-period average
# Short when price breaks below 4h Donchian lower channel AND 1d ATR ratio > 1.2 AND volume > 1.3x 20-period average
# Exit when price crosses 4h Donchian midpoint (mean reversion) OR 1d ATR ratio < 0.8 (low volatility)
# Uses 4h primary timeframe with 1d HTF for ATR volatility filter (more stable than same-TF ATR) and volume confirmation
# ATR ratio = current ATR / 20-period ATR average - identifies expanding volatility regimes
# Volume confirmation ensures breakouts have conviction
# Discrete sizing (0.25) to limit fee drag and manage drawdown
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe

name = "4h_Donchian20_Breakout_1dATR_Volume"
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
    
    # Get 1d data ONCE before loop for ATR volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ATR(14) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])
    
    # Wilder's ATR
    atr = np.full_like(tr, np.nan)
    for i in range(1, len(tr)):
        if i < 14:
            atr[i] = np.nan
        elif i == 14:
            atr[i] = np.nanmean(tr[1:15])
        else:
            if not np.isnan(atr[i-1]) and not np.isnan(tr[i]):
                atr[i] = (atr[i-1] * 13 + tr[i]) / 14
            else:
                atr[i] = np.nan
    
    # Calculate 20-period ATR average for ratio
    atr_ma_20 = pd.Series(atr).rolling(window=20, min_periods=20).mean().values
    atr_ratio = atr / atr_ma_20  # Current ATR / 20-period average ATR
    
    # Align ATR ratio to 4h timeframe
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # Calculate 4h Donchian channels (20-period)
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_upper + donchian_lower) / 2
    
    # Volume confirmation: volume > 1.3x 20-period average
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.3 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(atr_ratio_aligned[i]) or 
            np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or 
            np.isnan(donchian_mid[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian upper AND ATR expanding AND volume spike
            if (close[i] > donchian_upper[i] and 
                atr_ratio_aligned[i] > 1.2 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian lower AND ATR expanding AND volume spike
            elif (close[i] < donchian_lower[i] and 
                  atr_ratio_aligned[i] > 1.2 and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below Donchian midpoint (mean reversion) OR ATR contracting (low volatility)
            if close[i] < donchian_mid[i] or atr_ratio_aligned[i] < 0.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above Donchian midpoint (mean reversion) OR ATR contracting (low volatility)
            if close[i] > donchian_mid[i] or atr_ratio_aligned[i] < 0.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals