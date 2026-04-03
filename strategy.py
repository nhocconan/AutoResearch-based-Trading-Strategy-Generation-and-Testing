#!/usr/bin/env python3
"""
Experiment #191: 6h ADX + Williams Alligator Combination

HYPOTHESIS: Combining ADX trend strength with Williams Alligator (SMMA crossover) on 6h timeframe
provides robust trend-following signals. ADX > 25 filters for trending markets, while Alligator
jaw-teeth-lips alignment confirms direction. Uses 1d timeframe for HTF trend filter to avoid
counter-trend trades. Target: 75-150 total trades over 4 years (19-37/year) - within winning range.
Works in both bull and bear markets by only taking trades in direction of stronger trend (via HTF).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_adx_alligator_1d_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    close = prices["close"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for trend filter ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    # 1d EMA(50) for HTF trend direction
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    htf_trend_up = ema_50_1d > np.roll(ema_50_1d, 1)  # Rising EMA50 = uptrend
    htf_trend_down = ema_50_1d < np.roll(ema_50_1d, 1)  # Falling EMA50 = downtrend
    htf_trend_up_aligned = align_htf_to_ltf(prices, df_1d, htf_trend_up)
    htf_trend_down_aligned = align_htf_to_ltf(prices, df_1d, htf_trend_down)
    
    # === 6h Indicators ===
    # ADX(14) for trend strength
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    plus_dm = np.insert(plus_dm, 0, 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    minus_dm = np.insert(minus_dm, 0, 0)
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di_14 = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr_14
    minus_di_14 = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr_14
    dx = 100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Williams Alligator: SMMA(13,8), SMMA(8,5), SMMA(5,3)
    def smma(source, length):
        """Smoothed Moving Average"""
        sma = pd.Series(source).rolling(window=length, min_periods=length).mean().values
        smma_vals = np.full_like(source, np.nan, dtype=np.float64)
        smma_vals[length-1] = sma[length-1]
        for i in range(length, len(source)):
            if not np.isnan(sma[i]):
                smma_vals[i] = (smma_vals[i-1] * (length-1) + sma[i]) / length
            else:
                smma_vals[i] = smma_vals[i-1]
        return smma_vals
    
    jaw = smma(close, 13)  # Blue line
    teeth = smma(close, 8)  # Red line
    lips = smma(close, 5)   # Green line
    
    # Alligator signals: Lips > Teeth > Jaw = UP, Lips < Teeth < Jaw = DOWN
    alligator_up = (lips > teeth) & (teeth > jaw)
    alligator_down = (lips < teeth) & (teeth < jaw)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_bar = -1
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(adx[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or np.isnan(htf_trend_up_aligned[i]) or 
            np.isnan(htf_trend_down_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- ADX + Alligator Entry Logic ---
        # Strong trend (ADX > 25) + Alligator alignment + HTF trend filter
        strong_trend = adx[i] > 25
        
        # Long: ADX > 25 + Alligator bullish alignment + HTF uptrend
        long_condition = strong_trend & alligator_up[i] & htf_trend_up_aligned[i]
        # Short: ADX > 25 + Alligator bearish alignment + HTF downtrend
        short_condition = strong_trend & alligator_down[i] & htf_trend_down_aligned[i]
        
        # --- Position Management ---
        if in_position:
            # Check for trend weakening or Alligator reversal
            if position_side > 0:  # Long
                exit_condition = (adx[i] < 20) | ~alligator_up[i] | ~htf_trend_up_aligned[i]
            else:  # Short
                exit_condition = (adx[i] < 20) | ~alligator_down[i] | ~htf_trend_down_aligned[i]
            
            if exit_condition:
                in_position = False
                position_side = 0
                signals[i] = 0.0
            else:
                signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        if long_condition:
            in_position = True
            position_side = 1
            entry_price = close[i]
            signals[i] = SIZE
        elif short_condition:
            in_position = True
            position_side = -1
            entry_price = close[i]
            signals[i] = -SIZE
    
    return signals