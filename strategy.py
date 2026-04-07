#!/usr/bin/env python3
"""
1d_camarilla_pivot_1w_trend_volume_v2
Hypothesis: Daily Camarilla pivot levels with weekly trend filter and volume confirmation.
In bull markets: buy near L3 support in uptrend. In bear markets: sell near H3 resistance in downtrend.
Volume confirmation filters false breaks. Targets 10-25 trades/year by requiring pivot touch + volume spike + weekly trend alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_camarilla_pivot_1w_trend_volume_v2"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Daily data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for each day (using previous day's OHLC)
    # H4 = close + 1.5*(high-low), H3 = close + 1.0*(high-low), H2 = close + 0.5*(high-low)
    # L3 = close - 0.5*(high-low), L2 = close - 1.0*(high-low), L1 = close - 1.5*(high-low)
    high_low = high_1d - low_1d
    h4 = close_1d + 1.5 * high_low
    h3 = close_1d + 1.0 * high_low
    h2 = close_1d + 0.5 * high_low
    l3 = close_1d - 0.5 * high_low
    l2 = close_1d - 1.0 * high_low
    l1 = close_1d - 1.5 * high_low
    
    # Shift by 1 to use previous day's levels (no look-ahead)
    h4 = np.roll(h4, 1)
    h3 = np.roll(h3, 1)
    h2 = np.roll(h2, 1)
    l3 = np.roll(l3, 1)
    l2 = np.roll(l2, 1)
    l1 = np.roll(l1, 1)
    # First values will be incorrect but handled by validation
    
    # Align weekly trend to daily
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False).mean().values
    ema20_1d = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # 20-day volume average
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(ema20_1d[i]) or 
            np.isnan(h3[i]) or np.isnan(l3[i]) or
            np.isnan(vol_sma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        vol_confirm = volume[i] > 1.5 * vol_sma[i]
        
        if position == 1:  # Long position
            # Exit: price crosses below L3 OR trend turns down
            if close[i] < l3[i] or close[i] < ema20_1d[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price crosses above H3 OR trend turns up
            if close[i] > h3[i] or close[i] > ema20_1d[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: price touches L3 (support) in uptrend with volume
            if (abs(close[i] - l3[i]) < 0.001 * close[i] and  # Within 0.1% of L3
                vol_confirm and 
                close[i] > ema20_1d[i]):  # Weekly uptrend
                position = 1
                signals[i] = 0.25
            # Short: price touches H3 (resistance) in downtrend with volume
            elif (abs(close[i] - h3[i]) < 0.001 * close[i] and  # Within 0.1% of H3
                  vol_confirm and 
                  close[i] < ema20_1d[i]):  # Weekly downtrend
                position = -1
                signals[i] = -0.25
    
    return signals