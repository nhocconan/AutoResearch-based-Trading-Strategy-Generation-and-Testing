#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with volume confirmation and chop regime filter
# Uses 1d Camarilla levels (R1, S1) for structure, 12h for entry timing
# Volatility filter ensures trades only during sufficient movement
# Chop filter avoids whipsaws in ranging markets
# Designed to work in both bull (breakouts) and bear (breakdowns) markets
# Target: 50-150 trades over 4 years (12-37/year) to minimize fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivot levels (based on previous day)
    # Camarilla: R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    rang = prev_high - prev_low
    R1 = prev_close + 1.1 * rang / 12
    S1 = prev_close - 1.1 * rang / 12
    
    # Align Camarilla levels to 12h timeframe (already delayed by shift(1))
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # Calculate 12h ATR(14) for volatility filter
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.concatenate([[close[0]], close[:-1]])), 
                                          np.abs(low - np.concatenate([[close[0]], close[:-1]]))))
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 12h Choppiness Index(14) for regime filter
    # CHOP = 100 * log10(sum(ATR(14)) / log10(highest_high - lowest_low)) / log10(14)
    atr_sum = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / (hh - ll + 1e-10)) / np.log10(14)
    
    # Calculate 12h volume ratio (current vs 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / (vol_ma + 1e-10)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or 
            np.isnan(atr_14[i]) or np.isnan(chop[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: ATR > 0.5% of price
        vol_filter = atr_14[i] > 0.005 * close[i]
        
        # Chop regime filter: only trade when NOT excessively choppy (CHOP < 61.8)
        chop_filter = chop[i] < 61.8
        
        # Volume confirmation: volume > 1.5x average
        vol_confirm = vol_ratio[i] > 1.5
        
        # Long breakout: price > R1 with volume and volatility confirmation
        if (close[i] > R1_aligned[i] and vol_filter and chop_filter and vol_confirm):
            signals[i] = 0.25
            
        # Short breakdown: price < S1 with volume and volatility confirmation
        elif (close[i] < S1_aligned[i] and vol_filter and chop_filter and vol_confirm):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "12h_Camarilla_Pivot_Breakout_Volume_Chop_v1"
timeframe = "12h"
leverage = 1.0