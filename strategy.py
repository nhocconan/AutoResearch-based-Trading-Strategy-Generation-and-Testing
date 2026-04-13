#!/usr/bin/env python3
"""
6h_1d_ADX_Williams_Alligator
Hypothesis: Combine ADX trend strength with Williams Alligator to identify high-probability entries.
ADX > 25 indicates a trending market. Alligator jaw (SMA13) and teeth (SMA8) alignment confirms direction.
Use 1d ADX to filter trend strength on higher timeframe, and 6s Alligator for entry timing.
Works in both bull and bear by only taking trades in direction of strong trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_adx(high, low, close, period=14):
    """Calculate Average Directional Index."""
    plus_dm = np.diff(high, prepend=high[0])
    minus_dm = np.diff(low, prepend=low[0])
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)

    tr1 = np.abs(np.diff(high, prepend=high[0]))
    tr2 = np.abs(np.diff(low, prepend=low[0]))
    tr3 = np.abs(np.diff(close, prepend=close[0]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))

    atr = pd.Series(tr).rolling(window=period, min_periods=period).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=period, min_periods=period).mean().values / (atr + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).rolling(window=period, min_periods=period).mean().values / (atr + 1e-10)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).rolling(window=period, min_periods=period).mean().values
    return adx

def calculate_alligator(high, low, close):
    """Calculate Williams Alligator lines."""
    median_price = (high + low) / 2
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values  # SMA13
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values   # SMA8
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values    # SMA5
    return jaw, teeth, lips

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get daily data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d ADX for trend strength filter
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, period=14)
    strong_trend = adx_1d > 25
    
    # 6s Alligator for entry timing
    jaw, teeth, lips = calculate_alligator(high, low, close)
    
    # Align ADX to 6s timeframe
    strong_trend_aligned = align_htf_to_ltf(prices, df_1d, strong_trend.astype(float))
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25
    
    for i in range(50, n):
        # Skip if required data not ready
        if np.isnan(strong_trend_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]):
            signals[i] = 0.0
            continue
        
        # Long: strong trend + Alligator bullish alignment (lips > teeth > jaw)
        long_condition = strong_trend_aligned[i] > 0.5 and lips[i] > teeth[i] > jaw[i]
        
        # Short: strong trend + Alligator bearish alignment (lips < teeth < jaw)
        short_condition = strong_trend_aligned[i] > 0.5 and lips[i] < teeth[i] < jaw[i]
        
        if long_condition and position != 1:
            position = 1
            signals[i] = position_size
        elif short_condition and position != -1:
            position = -1
            signals[i] = -position_size
        else:
            # Hold or exit: exit when trend weakens or Alligator reverses
            if position == 1 and (strong_trend_aligned[i] <= 0.5 or lips[i] <= teeth[i]):
                position = 0
                signals[i] = 0.0
            elif position == -1 and (strong_trend_aligned[i] <= 0.5 or lips[i] >= teeth[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size if position == 1 else (-position_size if position == -1 else 0.0)
    
    return signals

name = "6h_1d_ADX_Williams_Alligator"
timeframe = "6h"
leverage = 1.0