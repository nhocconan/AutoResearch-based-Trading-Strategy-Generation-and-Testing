#!/usr/bin/env python3
"""
Hypothesis: 4h timeframe with 1d VWAP mean reversion and volume confirmation.
Price tends to revert to 1d VWAP after strong moves. Use 1d VWAP as dynamic support/resistance.
Enter long when price crosses above 1d VWAP with volume expansion, short when below.
Use 1w ADX to filter for trending vs ranging markets - only trade in ranging conditions (ADX < 25).
This should work in both bull (mean reversion in rallies) and bear (mean reversion in drops).
Target: 20-40 trades/year to avoid fee drag.
"""
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
    
    # Get 1d data for VWAP calculation
    df_1d = get_htf_data(prices, '1d')
    typical_price_1d = (df_1d['high'].values + df_1d['low'].values + df_1d['close'].values) / 3.0
    vwap_1d = (np.cumsum(typical_price_1d * df_1d['volume'].values) / 
               np.cumsum(df_1d['volume'].values))
    
    # Get 1w data for ADX filter (trend strength)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w ADX(14)
    plus_dm = np.where((high_1w[1:] - high_1w[:-1]) > (low_1w[:-1] - low_1w[1:]), 
                       np.maximum(high_1w[1:] - high_1w[:-1], 0), 0)
    minus_dm = np.where((low_1w[:-1] - low_1w[1:]) > (high_1w[1:] - high_1w[:-1]), 
                        np.maximum(low_1w[:-1] - low_1w[1:], 0), 0)
    tr = np.maximum(high_1w[1:] - low_1w[1:], 
                    np.maximum(np.abs(high_1w[1:] - close_1w[:-1]), 
                               np.abs(low_1w[1:] - close_1w[:-1])))
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean() / \
              pd.Series(tr).ewm(alpha=1/14, adjust=False).mean()
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean() / \
               pd.Series(tr).ewm(alpha=1/14, adjust=False).mean()
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx_1w = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean()
    # Prepend zero for first value
    adx_1w = np.concatenate([[0], adx_1w.values])
    
    # Align 1d VWAP and 1w ADX to 4h
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # Volume filter: current volume > 2.0x 24-period average (strong moves only)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_filter = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(vwap_1d_aligned[i]) or np.isnan(adx_1w_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Only trade in ranging markets (ADX < 25)
        if adx_1w_aligned[i] >= 25:
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price crosses above 1d VWAP with volume expansion
            if close[i] > vwap_1d_aligned[i] and close[i-1] <= vwap_1d_aligned[i-1] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price crosses below 1d VWAP with volume expansion
            elif close[i] < vwap_1d_aligned[i] and close[i-1] >= vwap_1d_aligned[i-1] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses back below 1d VWAP
            if close[i] < vwap_1d_aligned[i] and close[i-1] >= vwap_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses back above 1d VWAP
            if close[i] > vwap_1d_aligned[i] and close[i-1] <= vwap_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_1dVWAP_MeanReversion_Volume_ADXFilter"
timeframe = "4h"
leverage = 1.0