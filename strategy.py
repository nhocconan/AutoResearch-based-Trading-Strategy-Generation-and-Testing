#!/usr/bin/env python3
"""
Hypothesis: 6h Bollinger Band squeeze breakout with 1d ATR regime filter and volume confirmation.
Long when price breaks above upper BB AND 1d ATR ratio > 1.2 (low volatility regime) AND volume > 1.5x average.
Short when price breaks below lower BB AND 1d ATR ratio > 1.2 AND volume > 1.5x average.
Exit when price returns to middle BB OR ATR ratio drops below 0.8 (high volatility) OR volume < average.
Bollinger squeeze identifies low volatility breakouts. 1d ATR regime ensures trading only in favorable volatility conditions.
Volume confirmation avoids false breakouts. Designed for 6h timeframe targeting 50-150 total trades over 4 years.
Works in both bull and bear markets by capturing volatility expansion phases.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for ATR regime filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ATR(14) on 1d data
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align length
    atr14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate ATR(50) on 1d data for regime filter
    atr50_1d = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    
    # ATR ratio: short-term/long-term ATR (identifies volatility regimes)
    atr_ratio_1d = atr14_1d / atr50_1d
    
    # Align 1d ATR ratio to 6h timeframe
    atr_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio_1d)
    
    # Calculate Bollinger Bands(20,2) on 6h data
    sma20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_bb = sma20 + 2 * std20
    lower_bb = sma20 - 2 * std20
    middle_bb = sma20
    
    # Volume average (20-period) on primary timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(atr_ratio_1d_aligned[i]) or np.isnan(sma20[i]) or np.isnan(std20[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        atr_ratio = atr_ratio_1d_aligned[i]
        vol_ma_val = vol_ma[i]
        vol_current = volume[i]
        price = close[i]
        upper = upper_bb[i]
        lower = lower_bb[i]
        middle = middle_bb[i]
        
        if position == 0:
            # Long: Price breaks above upper BB AND low volatility regime (ATR ratio > 1.2) AND volume spike
            if (price > upper and atr_ratio > 1.2 and vol_current > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below lower BB AND low volatility regime (ATR ratio > 1.2) AND volume spike
            elif (price < lower and atr_ratio > 1.2 and vol_current > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Price returns to middle BB OR high volatility (ATR ratio < 0.8) OR volume drops
                if (price < middle or atr_ratio < 0.8 or vol_current < vol_ma_val):
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price returns to middle BB OR high volatility (ATR ratio < 0.8) OR volume drops
                if (price > middle or atr_ratio < 0.8 or vol_current < vol_ma_val):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_BBSqueeze_1dATRRegime_Volume"
timeframe = "6h"
leverage = 1.0