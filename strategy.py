#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1-week pivot points with volume confirmation
# - Primary signal: Price reacts at weekly pivot levels (R1, S1, R2, S2) with volume spike
# - Long: Price bounces off weekly S1/S2 with volume confirmation
# - Short: Price rejects at weekly R1/R2 with volume confirmation
# - Uses 1d ATR filter to avoid low volatility false signals
# - Position size: 0.25 discrete level to minimize fee churn
# - Stoploss: 1.5x ATR(14) on 6h timeframe
# - Target: 12-37 trades/year (50-150 total over 4 years) per 6h strategy guidelines
# - Works in bull/bear: Pivot points act as dynamic support/resistance in all markets

name = "6h_1w_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    if len(df_1w) < 10 or len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1-week pivot points (standard calculation)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    r1_1w = 2.0 * pivot_1w - low_1w
    s1_1w = 2.0 * pivot_1w - high_1w
    r2_1w = pivot_1w + (high_1w - low_1w)
    s2_1w = pivot_1w - (high_1w - low_1w)
    
    # Align weekly pivot levels to 6h timeframe (completed bar delay)
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    r2_1w_aligned = align_htf_to_ltf(prices, df_1w, r2_1w)
    s2_1w_aligned = align_htf_to_ltf(prices, df_1w, s2_1w)
    
    # Pre-compute 1d volume spike filter
    volume_1d = df_1d['volume'].values
    avg_volume_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_1d > (1.8 * avg_volume_20)  # Stricter threshold for fewer trades
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike)
    
    # Pre-compute 1d ATR(14) for volatility filter (avoid low volatility chop)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr_1d1 = high_1d - low_1d
    tr_1d2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr_1d3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr_1d1, np.maximum(tr_1d2, tr_1d3))
    tr_1d[0] = tr_1d1[0]
    atr_14 = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    atr_filter = (atr_14 / close_1d) > 0.015  # ATR > 1.5% of price (avoid low volatility)
    atr_filter_aligned = align_htf_to_ltf(prices, df_1d, atr_filter)
    
    # Pre-compute 6h ATR(14) for stoploss
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    
    tr_6h1 = high_6h - low_6h
    tr_6h2 = np.abs(high_6h - np.roll(close_6h, 1))
    tr_6h3 = np.abs(low_6h - np.roll(close_6h, 1))
    tr_6h = np.maximum(tr_6h1, np.maximum(tr_6h2, tr_6h3))
    tr_6h[0] = tr_6h1[0]
    atr_6h = pd.Series(tr_6h).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isclose(pivot_1w_aligned[i], 0) or np.isnan(vol_spike_aligned[i]) or 
            np.isnan(atr_filter_aligned[i]) or np.isnan(atr_6h[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Price reaches weekly R1 or stoploss hit
            if close_6h[i] >= r1_1w_aligned[i] or close_6h[i] < entry_price - 1.5 * atr_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price reaches weekly S1 or stoploss hit
            if close_6h[i] <= s1_1w_aligned[i] or close_6h[i] > entry_price + 1.5 * atr_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for pivot bounces/rejections with volume and volatility filters
            if vol_spike_aligned[i] and atr_filter_aligned[i]:
                # Long: Price bounces off weekly S1/S2 with volume
                if (close_6h[i] > s1_1w_aligned[i] and 
                    np.abs(close_6h[i] - s1_1w_aligned[i]) / s1_1w_aligned[i] < 0.005):  # Within 0.5% of S1
                    position = 1
                    entry_price = close_6h[i]
                    signals[i] = 0.25
                elif (close_6h[i] > s2_1w_aligned[i] and 
                      np.abs(close_6h[i] - s2_1w_aligned[i]) / s2_1w_aligned[i] < 0.005):  # Within 0.5% of S2
                    position = 1
                    entry_price = close_6h[i]
                    signals[i] = 0.25
                # Short: Price rejects at weekly R1/R2 with volume
                elif (close_6h[i] < r1_1w_aligned[i] and 
                      np.abs(close_6h[i] - r1_1w_aligned[i]) / r1_1w_aligned[i] < 0.005):  # Within 0.5% of R1
                    position = -1
                    entry_price = close_6h[i]
                    signals[i] = -0.25
                elif (close_6h[i] < r2_1w_aligned[i] and 
                      np.abs(close_6h[i] - r2_1w_aligned[i]) / r2_1w_aligned[i] < 0.005):  # Within 0.5% of R2
                    position = -1
                    entry_price = close_6h[i]
                    signals[i] = -0.25
    
    return signals