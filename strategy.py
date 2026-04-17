#!/usr/bin/env python3
"""
Hypothesis: 6h timeframe with 1d Camarilla pivot breakout + volume confirmation + ATR filter.
Long when price breaks above 1d Camarilla R3 with volume > 1.5x 20-period average and ATR(14) > 0.5 * ATR(50).
Short when price breaks below 1d Camarilla S3 with volume > 1.5x 20-period average and ATR(14) > 0.5 * ATR(50).
Use discrete position sizing of 0.25 to limit fee drag and manage drawdown.
Target: 50-150 total trades over 4 years (12-37/year) to avoid overtrading.
Camarilla pivots provide intraday support/resistance levels that work in ranging and trending markets.
Volume confirmation reduces false breakouts. ATR filter ensures sufficient volatility for meaningful moves.
Works in both bull and bear markets by trading breakouts in the direction of volatility expansion.
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
    
    # Get 1d data for Camarilla pivots and ATR
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d Camarilla Pivot Levels (based on previous day)
    # Camarilla levels: R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low), etc.
    # We use the previous day's range to calculate today's levels
    prev_close = np.concatenate([[close_1d[0]], close_1d[:-1]])
    prev_high = np.concatenate([[high_1d[0]], high_1d[:-1]])
    prev_low = np.concatenate([[low_1d[0]], low_1d[:-1]])
    
    # Camarilla multipliers
    camarilla_multiplier = 1.1 / 4  # R3/S3 level
    
    # Calculate R3 and S3 for each day (based on previous day's OHLC)
    camarilla_range = prev_high - prev_low
    camarilla_r3 = prev_close + camarilla_multiplier * camarilla_range * 4  # 1.1 * (high-low)
    camarilla_s3 = prev_close - camarilla_multiplier * camarilla_range * 4  # -1.1 * (high-low)
    
    # Calculate 1d ATR(14) and ATR(50) for volatility filter
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr3 = np.abs(low_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    
    # Calculate 1d volume 20-period average
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all to 6h
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    atr_50_aligned = align_htf_to_ltf(prices, df_1d, atr_50)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    vol_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(atr_14_aligned[i]) or np.isnan(atr_50_aligned[i]) or 
            np.isnan(vol_ma_20_aligned[i]) or np.isnan(vol_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.5x 20-period average
        volume_confirmed = vol_1d_aligned[i] > 1.5 * vol_ma_20_aligned[i]
        
        # ATR filter: short-term ATR > 50% of long-term ATR (volatility expansion)
        atr_expansion = atr_14_aligned[i] > 0.5 * atr_50_aligned[i]
        
        if position == 0:
            # Long: price breaks above Camarilla R3 with volume and volatility expansion
            if (close[i] > camarilla_r3_aligned[i] and 
                volume_confirmed and 
                atr_expansion):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla S3 with volume and volatility expansion
            elif (close[i] < camarilla_s3_aligned[i] and 
                  volume_confirmed and 
                  atr_expansion):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls below Camarilla S3 or volatility contracts
            if (close[i] < camarilla_s3_aligned[i] or 
                not atr_expansion):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises above Camarilla R3 or volatility contracts
            if (close[i] > camarilla_r3_aligned[i] or 
                not atr_expansion):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_1dCamarillaR3S3_Volume_ATR"
timeframe = "6h"
leverage = 1.0