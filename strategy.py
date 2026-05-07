# 4H_TRIX_Volume_Spike_Regime_v1
# Hypothesis: TRIX momentum combined with volume spike and chop regime filter (4h/1d).
# Long when TRIX crosses above zero with volume spike in chop regime.
# Short when TRIX crosses below zero with volume spike in chop regime.
# Exit when TRIX reverses or volume condition fails.
# Designed for low trade frequency (<50/year) with high win rate in trending markets.
# Works in both bull and bear markets via regime filter.
name = "4H_TRIX_Volume_Spike_Regime_v1"
timeframe = "4h"
leverage = 1.0

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
    
    # Get 1d data for chop regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d Chop Index (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # align with index
    
    # ATR
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Chop Index: (sum(TR) / (max(high) - min(low))) * 100
    max_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    range_max_min = max_high - min_low
    sum_tr = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    chop = (sum_tr / np.maximum(range_max_min, 1e-10)) * 100
    chop[range_max_min == 0] = 100  # avoid division by zero
    
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Get 4h data for TRIX
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 18:
        return np.zeros(n)
    
    # Calculate TRIX (12-period EMA of EMA of EMA)
    close_4h = pd.Series(df_4h['close'])
    ema1 = close_4h.ewm(span=12, adjust=False, min_periods=12).mean()
    ema2 = ema1.ewm(span=12, adjust=False, min_periods=12).mean()
    ema3 = ema2.ewm(span=12, adjust=False, min_periods=12).mean()
    trix = (ema3.pct_change() * 100).values  # percentage change
    
    trix_aligned = align_htf_to_ltf(prices, df_4h, trix)
    
    # Volume spike: current volume > 2.0 * 20-period average volume
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_avg * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(trix_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Chop regime: chop > 50 indicates ranging market (good for mean reversion)
        # But we use it as volatility filter: chop > 30 and chop < 70 for active market
        chop_ok = (chop_aligned[i] > 30) and (chop_aligned[i] < 70)
        
        if position == 0:
            # Long: TRIX crosses above zero with volume spike in active chop regime
            if (trix_aligned[i] > 0 and trix_aligned[i-1] <= 0 and 
                volume_spike[i] and chop_ok):
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses below zero with volume spike in active chop regime
            elif (trix_aligned[i] < 0 and trix_aligned[i-1] >= 0 and 
                  volume_spike[i] and chop_ok):
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: TRIX reverses or volume condition fails
            if position == 1 and (trix_aligned[i] < 0 or not volume_spike[i]):
                signals[i] = 0.0
                position = 0
            elif position == -1 and (trix_aligned[i] > 0 or not volume_spike[i]):
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals