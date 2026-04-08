#!/usr/bin/env python3
"""
4h_1d_vwap_std_dev_v1
Hypothesis: 4-hour strategy using 1-day VWAP with standard deviation bands.
Long when price touches VWAP - 1.5*std with volume > 1.5x average and price > 1d VWAP.
Short when price touches VWAP + 1.5*std with volume > 1.5x average and price < 1d VWAP.
Exit when price crosses VWAP or volume drops below 1.2x average.
Uses mean reversion to VWAP with volatility filtering, works in both trending and ranging markets.
Target: 20-40 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_vwap_std_dev_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1-day data for VWAP calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily VWAP and standard deviation
    typical_price_1d = (df_1d['high'].values + df_1d['low'].values + df_1d['close'].values) / 3.0
    vwap_numerator = np.cumsum(typical_price_1d * df_1d['volume'].values)
    vwap_denominator = np.cumsum(df_1d['volume'].values)
    vwap_1d = vwap_numerator / vwap_denominator
    
    # Calculate standard deviation of price from VWAP
    price_dev = typical_price_1d - vwap_1d
    # Rolling variance with min_periods=20
    var_1d = np.full(len(df_1d), np.nan)
    for i in range(20, len(df_1d)):
        var_1d[i] = np.var(price_dev[i-20:i])
    std_1d = np.sqrt(np.maximum(var_1d, 0))
    
    # VWAP bands: VWAP ± 1.5 * std
    upper_band_1d = vwap_1d + 1.5 * std_1d
    lower_band_1d = vwap_1d - 1.5 * std_1d
    
    # Align VWAP and bands to 4-hour timeframe
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    upper_band_1d_aligned = align_htf_to_ltf(prices, df_1d, upper_band_1d)
    lower_band_1d_aligned = align_htf_to_ltf(prices, df_1d, lower_band_1d)
    
    # Volume confirmation: 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(vwap_1d_aligned[i]) or np.isnan(upper_band_1d_aligned[i]) or 
            np.isnan(lower_band_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        price = close[i]
        vwap = vwap_1d_aligned[i]
        upper = upper_band_1d_aligned[i]
        lower = lower_band_1d_aligned[i]
        
        if position == 1:  # Long
            # Exit: price crosses above VWAP or volume drops below 1.2x average
            if price > vwap or vol_ratio < 1.2:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: price crosses below VWAP or volume drops below 1.2x average
            if price < vwap or vol_ratio < 1.2:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price touches lower band with volume expansion and below VWAP
            if price <= lower and vol_ratio > 1.5 and price < vwap:
                position = 1
                signals[i] = 0.25
            # Enter short: price touches upper band with volume expansion and above VWAP
            elif price >= upper and vol_ratio > 1.5 and price > vwap:
                position = -1
                signals[i] = -0.25
    
    return signals