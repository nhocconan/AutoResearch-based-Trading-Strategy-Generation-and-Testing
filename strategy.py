#!/usr/bin/env python3
"""
12h_Camarilla_R1S1_Breakout_VolumeFilter
Hypothesis: Camarilla pivot levels (R1, S1) from 1-day timeframe act as strong intraday support/resistance.
Price breaking above R1 or below S1 with volume confirmation indicates institutional interest and continuation.
Works in both bull and bear markets by only taking breakouts in the direction of the 1-day trend (EMA50).
Volume filter reduces false breakouts. Target: 15-25 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data once for Camarilla levels and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: based on previous day's range
    R1 = np.zeros_like(close_1d)
    S1 = np.zeros_like(close_1d)
    for i in range(len(close_1d)):
        if i == 0:
            R1[i] = np.nan
            S1[i] = np.nan
        else:
            range_val = high_1d[i-1] - low_1d[i-1]
            close_prev = close_1d[i-1]
            R1[i] = close_prev + (range_val * 1.1 / 12)
            S1[i] = close_prev - (range_val * 1.1 / 12)
    
    # 1-day EMA50 for trend filter
    close_series = pd.Series(close_1d)
    ema50_1d = close_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align to 12h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: current volume > 1.5x average volume of last 4 periods
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=4, min_periods=1).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if NaN in critical values
        if np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or np.isnan(ema50_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        price_open = prices['open'].iloc[i]
        R1_val = R1_aligned[i]
        S1_val = S1_aligned[i]
        ema50_val = ema50_1d_aligned[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Volume confirmation: at least 1.5x average volume
        vol_confirmed = vol_ratio >= 1.5
        
        if position == 0:
            # Long: price closes above R1 + volume confirmation + price above 1-day EMA50
            if price_close > R1_val and vol_confirmed and price_close > ema50_val:
                signals[i] = 0.25
                position = 1
            # Short: price closes below S1 + volume confirmation + price below 1-day EMA50
            elif price_close < S1_val and vol_confirmed and price_close < ema50_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price closes below S1 (reversal) or loses volume momentum
            if price_close < S1_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price closes above R1 (reversal) or loses volume momentum
            if price_close > R1_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_VolumeFilter"
timeframe = "12h"
leverage = 1.0