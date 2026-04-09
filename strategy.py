#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Camarilla pivot levels + volume confirmation + choppiness regime filter
# Long when price touches Camarilla L3 support with volume confirmation in trending regime (CHOP < 38.2)
# Short when price touches Camarilla H3 resistance with volume confirmation in trending regime
# Uses discrete position sizing 0.25 to target ~15-25 trades/year and minimize fee drag
# Works in bull/bear markets: pivot levels act as support/resistance, regime filter avoids ranging markets

name = "12h_1d_camarilla_pivot_v1"
timeframe = "12h"
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
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla pivot levels
    # Camarilla: H4 = close + 1.5*(high-low), H3 = close + 1.1*(high-low), L3 = close - 1.1*(high-low), L4 = close - 1.5*(high-low)
    range_1d = high_1d - low_1d
    camarilla_h3 = close_1d + 1.1 * range_1d
    camarilla_l3 = close_1d - 1.1 * range_1d
    
    # Align 1d Camarilla levels to 12h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Calculate 1d Choppiness Index (14-period) for regime filter
    # CHOP = 100 * log10(sum(ATR14) / (max(high14) - min(low14))) / log10(14)
    def true_range(h, l, c_prev):
        tr1 = h - l
        tr2 = np.abs(h - c_prev)
        tr3 = np.abs(l - c_prev)
        return np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Calculate ATR14
    tr_values = np.zeros(len(high_1d))
    tr_values[0] = high_1d[0] - low_1d[0]  # First TR is just high-low
    for i in range(1, len(high_1d)):
        tr_values[i] = true_range(high_1d[i], low_1d[i], close_1d[i-1])
    
    atr_14 = pd.Series(tr_values).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Choppiness Index
    max_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    sum_atr_14 = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    
    # Avoid division by zero
    denominator = max_high_14 - min_low_14
    chop = np.zeros(len(high_1d))
    mask = denominator != 0
    chop[mask] = 100 * np.log10(sum_atr_14[mask] / denominator[mask]) / np.log10(14)
    
    # Align Chop to 12h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Pre-compute volume confirmation (12h volume > 1.5x 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > 1.5 * vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(chop_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade in trending markets (CHOP < 38.2)
        if chop_aligned[i] >= 38.2:
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit long if price rises above Camarilla H3 (take profit) or falls below L3 (stop)
            if close[i] >= camarilla_h3_aligned[i] or close[i] <= camarilla_l3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit short if price falls below Camarilla L3 (take profit) or rises above H3 (stop)
            if close[i] <= camarilla_l3_aligned[i] or close[i] >= camarilla_h3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long at L3 support with volume confirmation
            if close[i] <= camarilla_l3_aligned[i] and volume_confirmed[i]:
                position = 1
                signals[i] = 0.25
            # Enter short at H3 resistance with volume confirmation
            elif close[i] >= camarilla_h3_aligned[i] and volume_confirmed[i]:
                position = -1
                signals[i] = -0.25
    
    return signals