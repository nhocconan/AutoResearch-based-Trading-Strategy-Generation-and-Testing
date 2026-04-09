#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Camarilla pivot levels with volume confirmation and choppiness regime filter
# Camarilla pivots from 1d provide key support/resistance levels for 4h timeframe
# Volume confirmation (current 4h volume > 1.8x 20-period average) filters false breakouts
# Choppiness regime filter: only trade when CHOP(14) > 61.8 (ranging market) for mean reversion at pivot levels
# Target: 20-50 trades/year on 4h timeframe (80-200 total over 4 years)
# Works in bull/bear: price reacts to 1d structure, volume confirms validity, chop filter avoids trending markets
# Discrete position sizing: 0.0, ±0.25 to minimize fee churn

name = "4h_1d_camarilla_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 25:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla pivot levels
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Key levels for trading: R3, R4, S3, S4 (stronger levels)
    camarilla_r3 = close_1d + range_1d * 1.1 / 4.0
    camarilla_r4 = close_1d + range_1d * 1.1 / 2.0
    camarilla_s3 = close_1d - range_1d * 1.1 / 4.0
    camarilla_s4 = close_1d - range_1d * 1.1 / 2.0
    
    # Align Camarilla levels to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Pre-compute volume confirmation (20-period average for 4h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute choppiness index (14-period) for regime filter
    # CHOP = 100 * log10(sum(ATR(1)) / (HHV(high,14) - LLV(low,14))) / log10(14)
    tr1 = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr1[0] = high[0] - low[0]  # First period TR
    atr_1 = pd.Series(tr1).rolling(window=1, min_periods=1).mean().values  # ATR(1) = TR
    sum_tr1 = pd.Series(tr1).rolling(window=14, min_periods=14).sum().values
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * (np.log10(sum_tr1 / (hh - ll)) / np.log10(14))
    # Handle division by zero and invalid values
    chop = np.where((hh - ll) == 0, 50.0, chop)
    chop = np.where(np.isnan(chop), 50.0, chop)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(r3_aligned[i]) or np.isnan(r4_aligned[i]) or
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(vol_ma_20[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > 1.8x average 4h volume
        volume_confirmed = volume[i] > 1.8 * vol_ma_20[i]
        
        # Regime filter: only trade in ranging markets (CHOP > 61.8)
        chop_filter = chop[i] > 61.8
        
        if position == 1:  # Long position
            # Exit on Camarilla S3 retracement (mean reversion from strong level)
            if close[i] < s3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit on Camarilla R3 retracement (mean reversion from strong level)
            if close[i] > r3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Mean reversion trading with volume and regime confirmation
            # Long on Camarilla S4 breakout (oversold bounce), Short on Camarilla R4 breakout (overbought rejection)
            if volume_confirmed and chop_filter:
                if close[i] < s4_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                elif close[i] > r4_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals