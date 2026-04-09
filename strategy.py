#!/usr/bin/env python3
# 4h_camarilla_vol_spike_v1
# Hypothesis: 4h mean reversion at daily Camarilla L3/H3 levels with volume spike confirmation.
# In ranging markets, price often reverts to mean after touching extreme daily levels (L3/H3).
# Volume spike confirms institutional participation at these levels, increasing probability of reversal.
# Works in both bull and bear markets as it fades extremes rather than following trends.
# Target: 20-50 trades/year (80-200 over 4 years) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_camarilla_vol_spike_v1"
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
    
    # Get 1d HTF data ONCE before loop for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels for daily
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Camarilla H3/L3 levels
    h3_1d = pivot_1d + (range_1d * 1.1 / 4)
    l3_1d = pivot_1d - (range_1d * 1.1 / 4)
    
    # Align Camarilla levels to 4h timeframe (completed daily candle only)
    h3_1d_aligned = align_htf_to_ltf(prices, df_1d, h3_1d)
    l3_1d_aligned = align_htf_to_ltf(prices, df_1d, l3_1d)
    
    # Volume spike detection (20-period volume average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma_20 * 2.0)
    
    # RSI(14) for overbought/oversold confirmation
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if any required data is NaN
        if (np.isnan(h3_1d_aligned[i]) or np.isnan(l3_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(rsi_values[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: RSI crosses above 50 or price reaches H3
            if rsi_values[i] > 50 or close[i] >= h3_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: RSI crosses below 50 or price reaches L3
            if rsi_values[i] < 50 or close[i] <= l3_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price touches or goes below L3 with RSI oversold and volume spike
            if (close[i] <= l3_1d_aligned[i]) and (rsi_values[i] < 30) and vol_spike[i]:
                position = 1
                signals[i] = 0.25
            # Enter short: price touches or goes above H3 with RSI overbought and volume spike
            elif (close[i] >= h3_1d_aligned[i]) and (rsi_values[i] > 70) and vol_spike[i]:
                position = -1
                signals[i] = -0.25
    
    return signals