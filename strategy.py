#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_Volume_Regime_v1
Hypothesis: Use daily Camarilla pivot levels (R1, S1) as key support/resistance with volume confirmation and chop regime filter.
Long when price breaks above R1 with volume > 1.5x average and chop < 61.8 (trending).
Short when price breaks below S1 with volume > 1.5x average and chop < 61.8.
Exit when price returns to the daily pivot (PP) or chop > 61.8 (choppy).
Targets 20-40 trades/year to avoid fee drag. Works in bull/bear via regime filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_hlf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === Daily Camarilla pivot levels ===
    df_1d = get_htf_data(prices, '1d')
    # Typical price for pivot calculation
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    # Calculate pivot (PP), support/resistance levels
    pp = typical_price.values
    r1 = pp + 1.1 * (df_1d['high'].values - df_1d['low'].values) / 12
    s1 = pp - 1.1 * (df_1d['high'].values - df_1d['low'].values) / 12
    
    # Align to 4h timeframe (wait for daily close)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # === Chop index (ETF-style) for regime filter ===
    def true_range(high, low, close_prev):
        return np.maximum(high - low, np.maximum(np.abs(high - close_prev), np.abs(low - close_prev)))
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Chop = 100 * log10(sum(TR14) / (max(high14) - min(low14))) / log10(14)
    def rolling_max(arr, window):
        return pd.Series(arr).rolling(window=window, min_periods=window).max().values
    
    def rolling_min(arr, window):
        return pd.Series(arr).rolling(window=window, min_periods=window).min().values
    
    sum_tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    max_high14 = rolling_max(high, 14)
    min_low14 = rolling_min(low, 14)
    
    # Avoid division by zero
    range14 = max_high14 - min_low14
    range14 = np.where(range14 == 0, 1e-10, range14)
    
    chop = 100 * np.log10(sum_tr14 / range14) / np.log10(14)
    
    # === Volume confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(pp_aligned[i]) or 
            np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(chop[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        # Regime filter: chop < 61.8 = trending (good for breakouts)
        trending_regime = chop[i] < 61.8
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price breaks above R1, volume confirmed, trending regime
            if (close[i] > r1_aligned[i] and 
                vol_confirmed and 
                trending_regime):
                signals[i] = 0.25
                position = 1
                continue
            # Short: price breaks below S1, volume confirmed, trending regime
            elif (close[i] < s1_aligned[i] and 
                  vol_confirmed and 
                  trending_regime):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic: return to pivot or choppy regime
        elif position == 1:
            # Exit long: price returns to PP OR chop > 61.8 (choppy)
            if (close[i] < pp_aligned[i] or 
                chop[i] > 61.8):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to PP OR chop > 61.8 (choppy)
            if (close[i] > pp_aligned[i] or 
                chop[i] > 61.8):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_Volume_Regime_v1"
timeframe = "4h"
leverage = 1.0