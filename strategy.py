#!/usr/bin/env python3
# 1d_WilliamsAlligator_Trend_Weekly_Slope
# Hypothesis: Williams Alligator (13,8,5 SMAs) on weekly timeframe determines trend; price above/below Alligator mouth on daily triggers entries. Works in bull (rides uptrend) and bear (avoids false longs in downtrend). Uses Williams %R for entry timing and volume confirmation. Target: 15-30 trades/year.

name = "1d_WilliamsAlligator_Trend_Weekly_Slope"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def sma(arr, period):
    """Simple moving average with NaN for insufficient data."""
    res = np.full_like(arr, np.nan, dtype=np.float64)
    if len(arr) >= period:
        for i in range(period - 1, len(arr)):
            res[i] = np.mean(arr[i - period + 1:i + 1])
    return res

def williams_r(high, low, close, period=14):
    """Williams %R: -100*(HHV - Close)/(HHV - LLV) over period."""
    res = np.full_like(close, np.nan, dtype=np.float64)
    if len(close) >= period:
        for i in range(period - 1, len(close)):
            hh = np.max(high[i - period + 1:i + 1])
            ll = np.min(low[i - period + 1:i + 1])
            if hh - ll != 0:
                res[i] = -100 * (hh - close[i]) / (hh - ll)
            else:
                res[i] = -50  # avoid division by zero
    return res

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly data for Alligator and Williams %R
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Williams Alligator: Jaw (13), Teeth (8), Lips (5) SMAs
    jaw = sma(close_1w, 13)
    teeth = sma(close_1w, 8)
    lips = sma(close_1w, 5)
    
    # Williams %R (14) for entry timing
    wr = williams_r(high_1w, low_1w, close_1w, 14)
    
    # Volume confirmation: 20-period average
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p - 1, len(arr)):
                res[i] = np.mean(arr[i - p + 1:i + 1])
        return res
    vol_ma_20 = mean_arr(volume_1w, 20)
    
    # Align all indicators to daily timeframe (wait for weekly bar to close)
    jaw_aligned = align_htf_to_ltf(prices, df_1w, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1w, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1w, lips)
    wr_aligned = align_htf_to_ltf(prices, df_1w, wr)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough history for indicators
    
    for i in range(start_idx, n):
        if np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or \
           np.isnan(lips_aligned[i]) or np.isnan(wr_aligned[i]) or \
           np.isnan(vol_ma_20_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine trend: Alligator alignment
        # Bullish: Lips > Teeth > Jaw (green alignment)
        # Bearish: Lips < Teeth < Jaw (red alignment)
        bullish_aligned = lips_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > jaw_aligned[i]
        bearish_aligned = lips_aligned[i] < teeth_aligned[i] and teeth_aligned[i] < jaw_aligned[i]
        
        if position == 0:
            # Long: bullish trend, price above Alligator mouth (Lips), Williams %R oversold, volume confirmation
            if bullish_aligned and close[i] > lips_aligned[i] and wr_aligned[i] < -80 and volume[i] > 1.5 * vol_ma_20_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: bearish trend, price below Alligator mouth (Lips), Williams %R overbought, volume confirmation
            elif bearish_aligned and close[i] < lips_aligned[i] and wr_aligned[i] > -20 and volume[i] > 1.5 * vol_ma_20_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: trend turns bearish or price crosses below Teeth
            if not bullish_aligned or close[i] < teeth_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: trend turns bullish or price crosses above Teeth
            if not bearish_aligned or close[i] > teeth_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals