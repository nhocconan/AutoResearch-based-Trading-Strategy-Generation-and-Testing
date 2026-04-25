#!/usr/bin/env python3
"""
6h ADX + Williams Alligator Combination
Hypothesis: ADX > 25 identifies trending markets while Williams Alligator (lips/teeth/jaw alignment) 
confirms trend direction and strength. Enter when ADX confirms trend and Alligator shows clear 
alignment (lips > teeth > jaw for long, lips < teeth < jaw for short). Exit when trend weakens 
(ADX < 20) or Alligator alignment breaks. Uses 1d timeframe for higher-context trend filter via 
EMA50 to avoid counter-trend trades. Designed for low frequency (12-37 trades/year) on 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for higher timeframe trend filter
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Williams Alligator on 6h timeframe
    # Jaw (blue line): 13-period SMMA of median price, shifted 8 bars
    # Teeth (red line): 8-period SMMA of median price, shifted 5 bars
    # Lips (green line): 5-period SMMA of median price, shifted 3 bars
    median_price = (high + low) / 2.0
    jaw_raw = pd.Series(median_price).rolling(window=13, min_periods=13).mean()
    teeth_raw = pd.Series(median_price).rolling(window=8, min_periods=8).mean()
    lips_raw = pd.Series(median_price).rolling(window=5, min_periods=5).mean()
    jaw = jaw_raw.shift(8).values
    teeth = teeth_raw.shift(5).values
    lips = lips_raw.shift(3).values
    
    # ADX calculation on 6h timeframe
    # +DM, -DM, TR
    up_move = high[1:] - high[:-1]
    down_move = low[:-1] - low[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[:-1] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Pad first element
    plus_dm = np.concatenate([[0.0], plus_dm])
    minus_dm = np.concatenate([[0.0], minus_dm])
    tr = np.concatenate([[0.0], tr])
    
    # Smoothed values (Wilder's smoothing = EMA with alpha=1/period)
    period = 14
    alpha = 1.0 / period
    atr = np.zeros_like(tr)
    atr[0] = tr[0]
    for i in range(1, len(tr)):
        atr[i] = alpha * tr[i] + (1 - alpha) * atr[i-1]
    
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=alpha, adjust=False).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=alpha, adjust=False).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(alpha=alpha, adjust=False).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for all indicators
    start_idx = max(13, 8, 5, 14) + max(8, 5, 3)  # Alligator shifts + ADX period
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or np.isnan(adx[i]) or np.isnan(plus_di[i]) or np.isnan(minus_di[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        
        # Trend filters
        htf_bullish = curr_close > ema_1d_aligned[i]
        htf_bearish = curr_close < ema_1d_aligned[i]
        strong_trend = adx[i] > 25
        weak_trend = adx[i] < 20  # exit threshold
        
        # Alligator alignment
        alligator_bullish = lips[i] > teeth[i] and teeth[i] > jaw[i]
        alligator_bearish = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        if position == 0:
            # Look for entry signals - require: strong trend + Alligator alignment + HTF bias
            long_entry = strong_trend and alligator_bullish and htf_bullish
            short_entry = strong_trend and alligator_bearish and htf_bearish
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: trend weakens OR Alligator alignment breaks OR loss of HTF bullish bias
            if weak_trend or (lips[i] < teeth[i]) or (curr_close < ema_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: trend weakens OR Alligator alignment breaks OR loss of HTF bearish bias
            if weak_trend or (lips[i] > teeth[i]) or (curr_close > ema_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ADX_WilliamsAlligator_Trend"
timeframe = "6h"
leverage = 1.0