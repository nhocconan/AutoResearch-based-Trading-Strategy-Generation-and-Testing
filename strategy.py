#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Choppiness_Adaptive_Breakout"
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
    
    # Get 1d data for Choppiness Index and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate Choppiness Index (14-day)
    # True Range
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = np.abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Sum of True Range over 14 periods
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    # Highest high and lowest low over 14 periods
    hh14 = df_1d['high'].rolling(window=14, min_periods=14).max().values
    ll14 = df_1d['low'].rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: 100 * log10(tr_sum / (hh14 - ll14)) / log10(14)
    chop = 100 * np.log10(tr_sum / (hh14 - ll14)) / np.log10(14)
    
    # Calculate ATR for volatility filtering
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Choppiness regime: > 61.8 = range (mean revert), < 38.2 = trending (trend follow)
    chop_regime = chop  # We'll use raw value for dynamic threshold
    
    # Get 4h data for price action
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Donchian channels (20-period)
    upper = df_4h['high'].rolling(window=20, min_periods=20).max().values
    lower = df_4h['low'].rolling(window=20, min_periods=20).min().values
    
    # Align all indicators to 4h timeframe
    chop_4h = align_htf_to_ltf(prices, df_1d, chop)
    atr_4h = align_htf_to_ltf(prices, df_1d, atr)
    upper_4h = align_htf_to_ltf(prices, df_4h, upper)
    lower_4h = align_htf_to_ltf(prices, df_4h, lower)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(chop_4h[i]) or np.isnan(atr_4h[i]) or 
            np.isnan(upper_4h[i]) or np.isnan(lower_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        chop_val = chop_4h[i]
        atr_val = atr_4h[i]
        upper_val = upper_4h[i]
        lower_val = lower_4h[i]
        
        # Dynamic entry threshold based on chop regime
        # In trending markets (chop < 38.2): breakout entries
        # In ranging markets (chop > 61.8): mean reversion at extremes
        if chop_val < 38.2:  # Trending regime
            if position == 0:
                # Enter long on break above upper Donchian
                if close[i] > upper_val:
                    signals[i] = 0.25
                    position = 1
                # Enter short on break below lower Donchian
                elif close[i] < lower_val:
                    signals[i] = -0.25
                    position = -1
            elif position == 1:
                # Exit long: close below midpoint of Donchian
                midpoint = (upper_val + lower_val) / 2
                if close[i] < midpoint:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: close above midpoint
                midpoint = (upper_val + lower_val) / 2
                if close[i] > midpoint:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
        else:  # Ranging regime (chop >= 38.2)
            if position == 0:
                # Enter long near lower Donchian (support)
                if close[i] < lower_val * 1.02:  # Within 2% of lower band
                    signals[i] = 0.25
                    position = 1
                # Enter short near upper Donchian (resistance)
                elif close[i] > upper_val * 0.98:  # Within 2% of upper band
                    signals[i] = -0.25
                    position = -1
            elif position == 1:
                # Exit long: close above midpoint or stop loss
                midpoint = (upper_val + lower_val) / 2
                if close[i] > midpoint:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: close below midpoint or stop loss
                midpoint = (upper_val + lower_val) / 2
                if close[i] < midpoint:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals