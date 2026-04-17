#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for ATR-based volatility filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily ATR(14)
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value has no previous close
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate daily volatility regime: ATR(14) / ATR(50)
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    atr_ratio = atr_14 / atr_50
    
    # Align ATR ratio to 12h timeframe
    atr_ratio_12h = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # Volume confirmation (20-period MA on 12h)
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # Need ATR(50) and volume MA20
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(volume_ma20[i]) or 
            np.isnan(atr_ratio_12h[i]) or
            np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: only trade when ATR ratio > 0.8 (avoid low volatility chop)
        vol_filter = atr_ratio_12h[i] > 0.8
        
        # Volume filter: current volume > 1.5x 20-period average
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        if position == 0:
            # Long: volatility expansion + volume spike
            if vol_filter and volume_filter:
                signals[i] = 0.25
                position = 1
            # Short: volatility expansion + volume spike (mean reversion in high vol)
            elif vol_filter and volume_filter and close[i] < np.mean(close[max(0, i-20):i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: volatility contraction or mean reversion
            if (atr_ratio_12h[i] < 0.6) or (close[i] < np.mean(close[max(0, i-10):i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: volatility contraction or mean reversion
            if (atr_ratio_12h[i] < 0.6) or (close[i] > np.mean(close[max(0, i-10):i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_VolatilityExpansion_Volume_MeanReversion"
timeframe = "12h"
leverage = 1.0