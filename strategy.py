#!/usr/bin/env python3
"""
12h_camarilla_pivot_1d_trend_volume_v2
Hypothesis: Price reverses at daily Camarilla pivot levels with volume confirmation and trend alignment.
Uses tighter entry conditions and adds daily ATR-based position sizing to reduce trade frequency
while maintaining edge in both bull and bear markets. Targets 12-37 trades per year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_1d_trend_volume_v2"
timeframe = "12h"
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
    
    # 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False).mean().values
    ema50_12h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 1d OHLC for Camarilla pivot calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    camarilla_range = 1.1 * (high_1d - low_1d) / 12
    camarilla_s1 = close_1d - camarilla_range  # Support level 1
    camarilla_r1 = close_1d + camarilla_range  # Resistance level 1
    
    # Align Camarilla levels to 12h timeframe
    camarilla_s1_12h = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    camarilla_r1_12h = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    
    # 20-period SMA for volume average
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Daily ATR for volatility filter
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    atr_12h = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(ema50_12h[i]) or 
            np.isnan(camarilla_s1_12h[i]) or 
            np.isnan(camarilla_r1_12h[i]) or 
            np.isnan(vol_sma[i]) or
            np.isnan(atr_12h[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        vol_confirm = volume[i] > 1.5 * vol_sma[i]
        
        # Volatility filter: avoid low volatility periods
        vol_filter = atr_12h[i] > 0.5 * np.nanmedian(atr_12h[max(0, i-50):i+1])
        
        if position == 1:  # Long position
            # Exit: price reaches resistance OR trend turns down
            if close[i] >= camarilla_r1_12h[i] or close[i] < ema50_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
        elif position == -1:  # Short position
            # Exit: price reaches support OR trend turns up
            if close[i] <= camarilla_s1_12h[i] or close[i] > ema50_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            # Long: price near support + volume confirmation + uptrend + volatility
            if (close[i] <= camarilla_s1_12h[i] * 1.005 and  # Allow small tolerance
                vol_confirm and 
                vol_filter and
                close[i] > ema50_12h[i]):
                position = 1
                signals[i] = 0.20
            # Short: price near resistance + volume confirmation + downtrend + volatility
            elif (close[i] >= camarilla_r1_12h[i] * 0.995 and  # Allow small tolerance
                  vol_confirm and 
                  vol_filter and
                  close[i] < ema50_12h[i]):
                position = -1
                signals[i] = -0.20
    
    return signals