#!/usr/bin/env python3
"""
Hypothesis:
12-hour Williams Alligator with 1-week trend filter and volume confirmation.
Williams Alligator uses three smoothed moving averages (Jaws, Teeth, Lips) to identify trends.
Only trade in direction of 1-week trend (price > weekly EMA50 for long, < for short).
Volume confirmation (current volume > 1.5x 20-period average) ensures momentum.
Designed for 12h timeframe to achieve 12-37 trades/year with low turnover.
Works in bull markets by catching trends and in bear markets by avoiding counter-trend trades via weekly filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_alligator(high, low, close):
    """Calculate Williams Alligator: Jaws(13,8), Teeth(8,5), Lips(5,3)"""
    # Typical price
    tp = (high + low + close) / 3.0
    
    # Smoothed Moving Average (SMMA) - same as RMA/Wilder's smoothing
    def smma(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            # First value is simple average
            result[period-1] = np.mean(data[:period])
            # Subsequent values: (prev * (period-1) + current) / period
            for i in range(period, len(data)):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    jaws = smma(tp, 13)  # Blue line
    teeth = smma(tp, 8)  # Red line
    lips = smma(tp, 5)   # Green line
    
    # Shift as per Williams: Jaws+8, Teeth+5, Lips+3
    jaws = np.roll(jaws, 8)
    teeth = np.roll(teeth, 5)
    lips = np.roll(lips, 3)
    
    return jaws, teeth, lips

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Williams Alligator
    jaws, teeth, lips = calculate_alligator(high, low, close)
    
    # === 1-week EMA50 (trend filter) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # EMA50 calculation
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: need enough data for Alligator (13+8 shift) and EMA50
    warmup = max(50, 20) + 13  # EMA50 + volatility + Alligator setup
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(jaws[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Alligator alignment: all three lines ordered
        # Bullish: Lips > Teeth > Jaws (green above red above blue)
        # Bearish: Lips < Teeth < Jaws (green below red below blue)
        bullish_alignment = lips[i] > teeth[i] and teeth[i] > jaws[i]
        bearish_alignment = lips[i] < teeth[i] and teeth[i] < jaws[i]
        
        # Trend filter: price relative to weekly EMA50
        price_above_ema50w = close[i] > ema50_1w_aligned[i]
        price_below_ema50w = close[i] < ema50_1w_aligned[i]
        
        # Volume confirmation
        vol_confirm = volume[i] > vol_ma_20[i] * 1.5
        
        # Entry logic: only enter when flat
        if position == 0:
            if bullish_alignment and price_above_ema50w and vol_confirm:
                signals[i] = 0.25
                position = 1
                continue
            elif bearish_alignment and price_below_ema50w and vol_confirm:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic: exit when Alligator alignment breaks or trend fails
        elif position == 1:
            # Exit long if bearish alignment forms or price drops below weekly EMA50
            if bearish_alignment or not price_above_ema50w:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short if bullish alignment forms or price rises above weekly EMA50
            if bullish_alignment or not price_below_ema50w:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsAlligator_1wEMA50_VolumeConfirm"
timeframe = "12h"
leverage = 1.0