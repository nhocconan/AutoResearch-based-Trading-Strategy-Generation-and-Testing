#!/usr/bin/env python3
"""
12h_Pivot_R1_S1_Breakout_Volume_ATRFilter_v1
Hypothesis: Trade Camarilla pivot breakouts on 12h timeframe with volume confirmation and ATR-based trend filter. 
Long when price breaks above R1 with volume > 1.5x average and ATR(12) > ATR(24) (trending up).
Short when price breaks below S1 with volume > 1.5x average and ATR(12) > ATR(24) (trending down).
Uses 1d Camarilla levels for structure and volatility filter to avoid whipsaw in ranging markets.
Designed for low trade frequency (15-25/year) to minimize fee impact while capturing strong trending moves.
Works in both bull and bear markets by following institutional pivot levels as support/resistance.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels (R1, S1) from previous day
    # R1 = close + 1.1*(high-low)/12
    # S1 = close - 1.1*(high-low)/12
    camarilla_r1 = np.full_like(close_1d, np.nan)
    camarilla_s1 = np.full_like(close_1d, np.nan)
    
    for i in range(1, len(close_1d)):
        prev_high = high_1d[i-1]
        prev_low = low_1d[i-1]
        prev_close = close_1d[i-1]
        camarilla_r1[i] = prev_close + 1.1 * (prev_high - prev_low) / 12
        camarilla_s1[i] = prev_close - 1.1 * (prev_high - prev_low) / 12
    
    # Align Camarilla levels to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # ATR for trend filter (ATR(12) > ATR(24) indicates trending market)
    def calculate_atr(high, low, close, period):
        tr = np.full_like(high, np.nan)
        for i in range(len(high)):
            if i == 0:
                tr[i] = high[i] - low[i]
            else:
                tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        atr = np.full_like(high, np.nan)
        if len(high) >= period:
            atr[period-1] = np.mean(tr[:period])
            for i in range(period, len(high)):
                atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        return atr
    
    atr_12 = calculate_atr(high, low, close, 12)
    atr_24 = calculate_atr(high, low, close, 24)
    
    # Volume confirmation: volume > 1.5x 24-period average
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 24
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(24, 24)  # Need enough data for ATR(24) and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(atr_12[i]) or np.isnan(atr_24[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Trend filter: ATR(12) > ATR(24) indicates trending market
        trending = atr_12[i] > atr_24[i]
        
        if position == 0:
            # Long: Price breaks above R1 + volume + trending
            if close[i] > r1_aligned[i] and vol_confirm and trending:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S1 + volume + trending
            elif close[i] < s1_aligned[i] and vol_confirm and trending:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price breaks below S1 or loss of trend
            if close[i] < s1_aligned[i] or not trending:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price breaks above R1 or loss of trend
            if close[i] > r1_aligned[i] or not trending:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Pivot_R1_S1_Breakout_Volume_ATRFilter_v1"
timeframe = "12h"
leverage = 1.0