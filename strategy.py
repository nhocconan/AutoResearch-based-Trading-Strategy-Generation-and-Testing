#!/usr/bin/env python3
"""
Hypothesis: 6h timeframe with 1d Camarilla pivot breakout + volume confirmation + ATR filter.
Long when price breaks above R3 with volume > 1.5x 20-period average and ATR(14) > ATR(30) MA (expanding volatility).
Short when price breaks below S3 with same conditions.
Camarilla pivots from daily timeframe provide institutional support/resistance levels.
Volume confirms participation, ATR filter ensures breakout occurs during expanding volatility (reduces false breakouts in chop).
Designed to work in trending markets (continuation breakouts) and avoid false signals in ranging markets.
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
    
    # Get 1d data for Camarilla pivots and volume
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d Camarilla pivot levels (R3, S3)
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # R3 = Close + Range * 1.1 / 2
    # S3 = Close - Range * 1.1 / 2
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    r3_1d = close_1d + range_1d * 1.1 / 2.0
    s3_1d = close_1d - range_1d * 1.1 / 2.0
    
    # Calculate 1d volume 20-period average
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate ATR(14) and ATR(30) for volatility filter
    def atr(high_vals, low_vals, close_vals, window):
        tr1 = np.abs(high_vals - low_vals)
        tr2 = np.abs(high_vals - np.roll(close_vals, 1))
        tr3 = np.abs(low_vals - np.roll(close_vals, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = 0  # first bar has no previous close
        atr_vals = pd.Series(tr).rolling(window=window, min_periods=window).mean().values
        return atr_vals
    
    atr_14 = atr(high, low, close, 14)
    atr_30 = atr(high, low, close, 30)
    atr_ratio = atr_14 / (atr_30 + 1e-10)  # avoid division by zero
    
    # Align all to primary timeframe (6h)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # need enough for ATR and volume averages
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r3_1d_aligned[i]) or 
            np.isnan(s3_1d_aligned[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i]) or 
            np.isnan(atr_ratio_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 1.5x 20-day average volume
        volume_confirmed = volume[i] > 1.5 * vol_ma_20_1d_aligned[i]
        # ATR filter: expanding volatility (ATR(14) > ATR(30))
        volatility_expanding = atr_ratio_aligned[i] > 1.0
        
        if position == 0:
            # Long: price breaks above R3 with volume and volatility expansion
            if (close[i] > r3_1d_aligned[i] and 
                volume_confirmed and 
                volatility_expanding):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 with volume and volatility expansion
            elif (close[i] < s3_1d_aligned[i] and 
                  volume_confirmed and 
                  volatility_expanding):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below R3 (failed breakout) or volatility contracts
            if close[i] < r3_1d_aligned[i] or not volatility_expanding:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above S3 (failed breakdown) or volatility contracts
            if close[i] > s3_1d_aligned[i] or not volatility_expanding:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_1dCamarilla_R3S3_Breakout_Volume_ATRFilter"
timeframe = "6h"
leverage = 1.0