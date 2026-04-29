#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h ADX(14) + Williams Alligator combination with 1d trend filter
# ADX > 25 indicates trending market, Alligator lines (Jaw/Teeth/Lips) provide entry/exit signals
# 1d EMA50 trend filter ensures alignment with higher timeframe direction
# Works in both bull/bear markets: ADX filters range conditions, Alligator catches trends
# Target: 60-120 total trades over 4 years (15-30/year) on 6h timeframe

name = "6h_ADX_Alligator_1dEMA50_TrendFilter_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams Alligator on 6h timeframe (SMAs with specific periods)
    # Jaw: 13-period SMA shifted 8 bars ahead
    # Teeth: 8-period SMA shifted 5 bars ahead  
    # Lips: 5-period SMA shifted 3 bars ahead
    close_s = pd.Series(close)
    jaw = close_s.rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = close_s.rolling(window=8, min_periods=8).mean().shift(5).values
    lips = close_s.rolling(window=5, min_periods=5).mean().shift(3).values
    
    # ADX(14) calculation
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar has no previous close
    
    # Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed values (Wilder's smoothing = EMA with alpha=1/period)
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 21, 14, 13)  # Warmup for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or np.isnan(adx[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_ema_1d = ema_50_1d_aligned[i]
        curr_jaw = jaw[i]
        curr_teeth = teeth[i]
        curr_lips = lips[i]
        curr_adx = adx[i]
        
        # Alligator alignment: Lips > Teeth > Jaw = bullish, Lips < Teeth < Jaw = bearish
        bullish_alignment = curr_lips > curr_teeth and curr_teeth > curr_jaw
        bearish_alignment = curr_lips < curr_teeth and curr_teeth < curr_jaw
        
        # ADX filter: only trade when trending (ADX > 25)
        trending = curr_adx > 25
        
        # Handle exits
        if position == 1:  # Long position
            # Exit: bearish Alligator alignment OR ADX drops below 20 (trend weakening)
            if bearish_alignment or curr_adx < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: bullish Alligator alignment OR ADX drops below 20
            if bullish_alignment or curr_adx < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: bullish Alligator alignment + price above 1d EMA50 + ADX > 25
            if bullish_alignment and curr_close > curr_ema_1d and trending:
                signals[i] = 0.25
                position = 1
            # Short entry: bearish Alligator alignment + price below 1d EMA50 + ADX > 25
            elif bearish_alignment and curr_close < curr_ema_1d and trending:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals