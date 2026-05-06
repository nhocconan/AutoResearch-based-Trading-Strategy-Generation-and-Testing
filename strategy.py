#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1-day Choppiness Index and 4-period RSI for mean reversion
# - Uses daily Choppiness Index > 61.8 to identify ranging markets
# - In ranging markets, enters long when 4-period RSI < 30 and short when > 70
# - Exits when RSI crosses back above 50 (long) or below 50 (short)
# - Designed to capture mean reversion in ranging markets while avoiding trends
# - Target: 50-150 total trades over 4 years (12-37/year) with 0.25 position sizing

name = "4h_ChoppinessRSI_MeanReversion"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for Choppiness Index calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 14-period Choppiness Index
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # ATR(14)
    atr = np.convolve(tr, np.ones(14)/14, mode='full')[:len(tr)]
    atr[:13] = np.nan  # First 13 values invalid
    
    # Sum of TR over 14 periods
    sum_tr = np.convolve(tr, np.ones(14), mode='full')[:len(tr)]
    sum_tr[:13] = np.nan
    
    # Highest high and lowest low over 14 periods
    hh = np.concatenate([[np.nan]*13, np.maximum.accumulate(high_1d)[13:]])
    ll = np.concatenate([[np.nan]*13, np.minimum.accumulate(low_1d)[13:]])
    # Fix the first values
    hh[:14] = np.nan
    ll[:14] = np.nan
    for i in range(14, len(high_1d)):
        hh[i] = np.max(high_1d[i-13:i+1])
        ll[i] = np.min(low_1d[i-13:i+1])
    
    # Choppiness Index formula: 100 * log10(sum_tr / (hh - ll)) / log10(14)
    # Avoid division by zero
    hh_ll = hh - ll
    ci = np.where((hh_ll > 0) & (~np.isnan(sum_tr)), 
                  100 * np.log10(sum_tr / hh_ll) / np.log10(14), 
                  np.nan)
    
    # Align Choppiness Index to 4h timeframe
    ci_4h = align_htf_to_ltf(prices, df_1d, ci)
    
    # Calculate 4-period RSI on 4h timeframe
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.convolve(gain, np.ones(4)/4, mode='full')[:len(gain)]
    avg_loss = np.convolve(loss, np.ones(4)/4, mode='full')[:len(loss)]
    avg_gain[:3] = np.nan
    avg_loss[:3] = np.nan
    
    # Calculate RSI
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi[:3] = np.nan  # First 3 values invalid
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup
        # Skip if any critical value is NaN
        if (np.isnan(ci_4h[i]) or np.isnan(rsi[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Only trade in ranging markets (Choppiness > 61.8)
            if ci_4h[i] > 61.8:
                # Long when RSI < 30 (oversold)
                if rsi[i] < 30:
                    signals[i] = 0.25
                    position = 1
                # Short when RSI > 70 (overbought)
                elif rsi[i] > 70:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long when RSI crosses back above 50
            if rsi[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short when RSI crosses back below 50
            if rsi[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals