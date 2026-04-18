#!/usr/bin/env python3
"""
Strategy: 1d Williams Alligator with 1w Trend Filter and Volume Confirmation
Hypothesis: Williams Alligator (Jaw/Teeth/Lips) identifies market structure on 1d timeframe.
When price is aligned (above all lines in uptrend, below all lines in downtrend) with volume confirmation,
it indicates strong trend continuation. The 1w EMA(50) acts as a higher timeframe trend filter to avoid
counter-trend trades. Designed for 15-25 trades/year to minimize fee drag.
Works in bull markets (buy when aligned above) and bear markets (sell when aligned below).
Uses Williams Alligator formula: Jaw=SMMA(13,8), Teeth=SMMA(8,5), Lips=SMMA(5,3)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(data, period):
    """Smoothed Moving Average (SMMA) - same as RMA/Wilder's"""
    if len(data) < period:
        return np.full(len(data), np.nan)
    result = np.full(len(data), np.nan)
    result[period-1] = np.mean(data[:period])
    for i in range(period, len(data)):
        result[i] = (result[i-1] * (period-1) + data[i]) / period
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams Alligator
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Typical price for Alligator
    typical_price_1d = (high_1d + low_1d + close_1d) / 3.0
    
    # Williams Alligator lines
    jaw = smma(typical_price_1d, 13)  # Blue line (13,8)
    teeth = smma(typical_price_1d, 8)  # Red line (8,5)
    lips = smma(typical_price_1d, 5)   # Green line (5,3)
    
    # Align Alligator lines to 1d timeframe (no additional delay needed for SMMA)
    jaw_1d = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_1d = align_htf_to_ltf(prices, df_1d, teeth)
    lips_1d = align_htf_to_ltf(prices, df_1d, lips)
    
    # Get 1w data for EMA(50) trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate EMA(50) on 1w close
    ema_50_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 50:
        ema_50_1w[49] = np.mean(close_1w[:50])
        for i in range(50, len(close_1w)):
            ema_50_1w[i] = (close_1w[i] * 2/51) + (ema_50_1w[i-1] * 49/51)
    
    # Align 1w EMA to 1d timeframe
    ema_50_1w_1d = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # need EMA50, volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(jaw_1d[i]) or np.isnan(teeth_1d[i]) or np.isnan(lips_1d[i]) or 
            np.isnan(ema_50_1w_1d[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3 * 20-period average
        vol_confirmed = volume[i] > 1.3 * vol_ma[i]
        
        # Alligator alignment: Lips > Teeth > Jaw (uptrend) or Lips < Teeth < Jaw (downtrend)
        aligned_up = lips_1d[i] > teeth_1d[i] and teeth_1d[i] > jaw_1d[i]
        aligned_down = lips_1d[i] < teeth_1d[i] and teeth_1d[i] < jaw_1d[i]
        
        # Higher timeframe trend filter
        trend_up = close[i] > ema_50_1w_1d[i]
        trend_down = close[i] < ema_50_1w_1d[i]
        
        if position == 0:
            # Long entry: Alligator aligned up + volume + 1w uptrend
            if aligned_up and vol_confirmed and trend_up:
                signals[i] = 0.25
                position = 1
            # Short entry: Alligator aligned down + volume + 1w downtrend
            elif aligned_down and vol_confirmed and trend_down:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: Alligator alignment breaks or reverse signal
            if not aligned_up:  # Lips <= Teeth or Teeth <= Jaw
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Alligator alignment breaks or reverse signal
            if not aligned_down:  # Lips >= Teeth or Teeth >= Jaw
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WilliamsAlligator_1wEMA50_Volume"
timeframe = "1d"
leverage = 1.0