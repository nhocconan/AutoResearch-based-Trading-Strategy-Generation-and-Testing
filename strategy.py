#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Choppiness Index (CI) regime filter with 1-day EMA trend and volume confirmation
# Choppiness Index (CI) measures whether the market is trending (CI < 38.2) or ranging (CI > 61.8).
# We go long when CI indicates ranging AND price touches 1-day EMA(34) support with volume spike,
# and short when CI indicates ranging AND price touches 1-day EMA(34) resistance with volume spike.
# This strategy avoids whipsaws in trending markets by only trading in ranging conditions.
# Target: 50-150 total trades over 4 years = 12-37/year on 12h timeframe.

name = "12h_Choppiness_1dEMA34_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data once
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate daily EMA(34) for trend reference
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate Choppiness Index (CI) on 12h data (period=14)
    # CI = 100 * log10(sum(ATR over n periods) / (log10(highest high - lowest low) * n))
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    range_hl = highest_high - lowest_low
    
    # Avoid division by zero
    ci_raw = np.where(range_hl > 0, atr_sum / range_hl, 1.0)
    ci = 100 * np.log10(ci_raw)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(ci[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(range_hl[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema34_1d_val = ema34_1d_aligned[i]
        ci_val = ci[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: ranging market (high CI) + price at EMA support + volume spike
            if (ci_val > 61.8 and 
                low[i] <= ema34_1d_val * 1.002 and  # within 0.2% of EMA as support
                vol_spike):
                signals[i] = 0.25
                position = 1
            # Enter short: ranging market (high CI) + price at EMA resistance + volume spike
            elif (ci_val > 61.8 and 
                  high[i] >= ema34_1d_val * 0.998 and  # within 0.2% of EMA as resistance
                  vol_spike):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: market starts trending (low CI) OR price moves significantly above EMA
            if (ci_val < 38.2 or close[i] > ema34_1d_val * 1.01):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: market starts trending (low CI) OR price moves significantly below EMA
            if (ci_val < 38.2 or close[i] < ema34_1d_val * 0.99):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals