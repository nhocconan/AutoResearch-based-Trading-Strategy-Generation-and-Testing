#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + volume confirmation + ATR regime filter
# Long when: Price breaks above Donchian upper (20) AND volume > 1.5x 20-period average volume AND ATR(14) < ATR(50) (low volatility regime)
# Short when: Price breaks below Donchian lower (20) AND volume > 1.5x 20-period average volume AND ATR(14) < ATR(50) (low volatility regime)
# Exit when price returns to Donchian middle (mean of upper and lower)
# Donchian breakout captures volatility expansion after consolidation
# Volume confirmation ensures breakout is supported by participation
# ATR regime filter ensures we only trade during low volatility periods (pre-breakout squeeze)
# Works in both bull and bear markets by trading breakouts in direction of the squeeze break
# Target: 75-200 total trades over 4 years (19-50/year) with discrete sizing 0.25

name = "4h_DonchianBreakout_Volume_ATRRegime"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for ATR regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need enough for ATR(50)
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d True Range and ATR
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan  # First value has no previous close
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50_1d = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    atr_50_aligned = align_htf_to_ltf(prices, df_1d, atr_50_1d)
    
    # Calculate Donchian Channels (20) on 4h
    upper_donchian = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lower_donchian = pd.Series(low).rolling(window=20, min_periods=20).min().values
    middle_donchian = (upper_donchian + lower_donchian) / 2.0
    
    # Calculate volume confirmation: volume > 1.5x 20-period average volume
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = avg_volume * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(atr_14_aligned[i]) or np.isnan(atr_50_aligned[i]) or 
            np.isnan(upper_donchian[i]) or np.isnan(lower_donchian[i]) or 
            np.isnan(avg_volume[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # ATR regime: low volatility (ATR14 < ATR50)
        low_vol_regime = atr_14_aligned[i] < atr_50_aligned[i]
        # Volume confirmation
        volume_confirmed = volume[i] > volume_threshold[i]
        
        if position == 0:
            # Long: Break above upper Donchian in low volatility regime with volume
            if close[i] > upper_donchian[i] and low_vol_regime and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: Break below lower Donchian in low volatility regime with volume
            elif close[i] < lower_donchian[i] and low_vol_regime and volume_confirmed:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: return to middle Donchian (mean reversion)
            if close[i] < middle_donchian[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: return to middle Donchian (mean reversion)
            if close[i] > middle_donchian[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals