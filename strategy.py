#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot reversals with volume confirmation and volatility filter.
# Works in bull (breakouts above R3) and bear (reversals at S3) via mean reversion at extreme levels.
# Low turnover: only triggers at statistically significant pivot levels with volume surge.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily Camarilla pivot levels (based on prior day OHLC)
    df_1d = get_htf_data(prices, '1d')
    # Calculate from prior day's close to avoid look-ahead (use shift(1))
    c = df_1d['close'].shift(1).values
    h = df_1d['high'].shift(1).values
    l = df_1d['low'].shift(1).values
    
    # Camarilla levels
    rng = h - l
    h3 = c + (rng * 1.1 / 4)
    l3 = c - (rng * 1.1 / 4)
    h4 = c + (rng * 1.1 / 2)
    l4 = c - (rng * 1.1 / 2)
    
    # Align to 4h (wait for daily close)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    
    # Volume confirmation: current > 2.0x median of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=20).median()
    vol_threshold = 2.0 * vol_median
    
    # Volatility filter: avoid choppy markets
    atr = pd.Series(np.maximum(high - low,
                               np.maximum(np.abs(high - np.concatenate([[high[0]], high[:-1]])),
                                          np.abs(low - np.concatenate([[low[0]], low[:-1]]))))).rolling(14, min_periods=14).mean()
    atr_median = atr.rolling(window=50, min_periods=50).median()
    vol_filter = atr > 0.5 * atr_median  # Require sufficient volatility
    
    signals = np.zeros(n)
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(vol_threshold[i]) or np.isnan(vol_filter[i])):
            continue
        
        # Long: price at S3/S4 with volume spike (mean reversion long)
        if close[i] <= l3_aligned[i] and volume[i] > vol_threshold[i] and vol_filter[i]:
            signals[i] = 0.25
        
        # Short: price at R3/R4 with volume spike (mean reversion short)
        elif close[i] >= h3_aligned[i] and volume[i] > vol_threshold[i] and vol_filter[i]:
            signals[i] = -0.25
        
        # Exit: price returns to mean (middle of range)
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and close[i] > (l3_aligned[i] + h3_aligned[i]) / 2) or
               (signals[i-1] == -0.25 and close[i] < (l3_aligned[i] + h3_aligned[i]) / 2))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "4h_Camarilla_MeanReversion_Volume"
timeframe = "4h"
leverage = 1.0