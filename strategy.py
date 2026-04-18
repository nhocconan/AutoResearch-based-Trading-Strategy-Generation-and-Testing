#!/usr/bin/env python3
"""
4h_Donchian_Breakout_With_Volume_and_ADX_Trend_Filter
Hypothesis: Buy when price breaks above Donchian upper band (20) with volume surge and strong ADX trend; short when breaks below lower band. Donchian channels capture breakouts effectively in both trending and ranging markets. Volume confirms institutional participation, and ADX ensures we only trade during strong trends, reducing whipsaw in sideways markets. Designed for low trade frequency (<30/year) to minimize fee drag while capturing high-probability trend continuations.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike: >2.5x 20-period average (stricter to reduce trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.5 * vol_ma)
    
    # ADX trend filter (14-period) - only trade when trend is strong
    # Calculate +DM, -DM, TR
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    close_series = pd.Series(close)
    
    up_move = high_series.diff()
    down_move = low_series.diff().multiply(-1)
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    tr1 = high_series - low_series
    tr2 = abs(high_series - close_series.shift())
    tr3 = abs(low_series - close_series.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Smoothed values
    atr = tr.rolling(window=14, min_periods=14).mean().values
    plus_dm_smooth = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values
    
    # DI values
    plus_di = 100 * plus_dm_smooth / atr
    minus_di = 100 * minus_dm_smooth / atr
    
    # DX and ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Strong trend filter: ADX > 25
    strong_trend = adx > 25
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 40  # Need all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(high_roll[i]) or 
            np.isnan(low_roll[i]) or
            np.isnan(volume_spike[i]) or
            np.isnan(strong_trend[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper = high_roll[i]
        lower = low_roll[i]
        vol_spike = volume_spike[i]
        trend_strong = strong_trend[i]
        
        if position == 0:
            # Long: price breaks above upper band with volume surge and strong trend
            if price > upper and vol_spike and trend_strong:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower band with volume surge and strong trend
            elif price < lower and vol_spike and trend_strong:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price breaks below lower band or trend weakens
            if price < lower or not trend_strong:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price breaks above upper band or trend weakens
            if price > upper or not trend_strong:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Donchian_Breakout_With_Volume_and_ADX_Trend_Filter"
timeframe = "4h"
leverage = 1.0