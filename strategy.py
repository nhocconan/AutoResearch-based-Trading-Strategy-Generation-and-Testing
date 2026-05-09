#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1-day Choppiness Index regime filter (CHOP > 61.8 = range, CHOP < 38.2 = trend)
# Combined with 1-day Bollinger Band mean reversion: long when price < lower BB and CHOP > 61.8,
# short when price > upper BB and CHOP > 61.8. Uses 1-day indicators to avoid look-ahead.
# Designed to work in both bull and bear markets by focusing on mean reversion in ranging markets.
# Target: 20-50 trades/year to minimize fee drag.

name = "12h_Chop_BB_MeanReversion"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Choppiness Index and Bollinger Bands
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1-day Choppiness Index (CHOP)
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
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Highest high and lowest low over 14 periods
    hh_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Chop = 100 * log10(sum(TR14) / (HH14 - LL14)) / log10(14)
    sum_tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    hh_ll_diff = hh_14 - ll_14
    chop = 100 * np.log10(sum_tr14 / hh_ll_diff) / np.log10(14)
    chop = np.where(hh_ll_diff == 0, 100, chop)  # Avoid division by zero
    chop = np.where(sum_tr14 == 0, 0, chop)
    
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Calculate 1-day Bollinger Bands (20, 2)
    sma_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + (2 * std_20)
    lower_bb = sma_20 - (2 * std_20)
    
    upper_bb_aligned = align_htf_to_ltf(prices, df_1d, upper_bb)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1d, lower_bb)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Need enough data for CHOP and BB calculations
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(chop_aligned[i]) or 
            np.isnan(upper_bb_aligned[i]) or 
            np.isnan(lower_bb_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        chop_val = chop_aligned[i]
        upper_bb_val = upper_bb_aligned[i]
        lower_bb_val = lower_bb_aligned[i]
        
        if position == 0:
            # Enter long: Price below lower BB and choppy market (CHOP > 61.8)
            if close[i] < lower_bb_val and chop_val > 61.8:
                signals[i] = 0.25
                position = 1
            # Enter short: Price above upper BB and choppy market (CHOP > 61.8)
            elif close[i] > upper_bb_val and chop_val > 61.8:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price crosses above SMA or market starts trending (CHOP < 38.2)
            if close[i] > sma_20[-1] if len(sma_20) > 0 else False or chop_val < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price crosses below SMA or market starts trending (CHOP < 38.2)
            if close[i] < sma_20[-1] if len(sma_20) > 0 else False or chop_val < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals