#!/usr/bin/env python3
"""
1d_WilliamsAlligator_Follow_1wTrend
Hypothesis: Use Williams Alligator (3 SMAs: Jaw=13, Teeth=8, Lips=5) on 1d to detect trend direction. Go long when price > Teeth and Lips > Jaw (bullish alignment), short when price < Teeth and Lips < Jaw (bearish alignment). Filter with 1w ADX > 25 to ensure strong trend. Exit when Alligator alignment breaks or price crosses Jaw. Designed for low trade frequency (<20/year) to avoid fee drag while capturing sustained trends in both bull and bear markets.
"""

name = "1d_WilliamsAlligator_Follow_1wTrend"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for Williams Alligator
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Get 1w data for ADX trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # 1d OHLCV
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Williams Alligator: SMAs of median price (HL/2)
    median_price = (high_1d + low_1d) / 2.0
    
    # Jaw: 13-period SMMA (smoothed with 8-period offset)
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean()
    jaw = jaw.rolling(window=8, min_periods=8).mean()  # SMMA via double smoothing
    jaw = jaw.values
    
    # Teeth: 8-period SMMA (smoothed with 5-period offset)
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean()
    teeth = teeth.rolling(window=5, min_periods=5).mean()
    teeth = teeth.values
    
    # Lips: 5-period SMMA (smoothed with 3-period offset)
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean()
    lips = lips.rolling(window=3, min_periods=3).mean()
    lips = lips.values
    
    # 1w ADX for trend filter (14 period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = np.abs(high_1w - low_1w)
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_1w = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Directional Movement
    up_move = np.diff(high_1w, prepend=high_1w[0])
    down_move = -np.diff(low_1w, prepend=low_1w[0])
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed DM
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr_1w
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr_1w
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # Align Alligator lines to 1d
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start after warmup period
    start_idx = 40  # for Alligator and ADX
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(adx_1w_aligned[i])):
            if position != 0:
                # Exit if alignment breaks
                if position == 1 and not (close_1d[i] > teeth_aligned[i] and lips_aligned[i] > jaw_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                elif position == -1 and not (close_1d[i] < teeth_aligned[i] and lips_aligned[i] < jaw_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25 if position == 1 else -0.25
            continue
        
        # Trend filter: only trade when 1w ADX > 25
        strong_trend = adx_1w_aligned[i] > 25
        
        if position == 0:
            # Look for entries based on Alligator alignment
            if strong_trend:
                # Bullish alignment: price > Teeth and Lips > Jaw
                if close_1d[i] > teeth_aligned[i] and lips_aligned[i] > jaw_aligned[i]:
                    signals[i] = 0.25  # long
                    position = 1
                    entry_price = close_1d[i]
                # Bearish alignment: price < Teeth and Lips < Jaw
                elif close_1d[i] < teeth_aligned[i] and lips_aligned[i] < jaw_aligned[i]:
                    signals[i] = -0.25  # short
                    position = -1
                    entry_price = close_1d[i]
        else:
            # Manage existing position: exit when alignment breaks
            if position == 1:
                # Long: exit when price <= Teeth or Lips <= Jaw
                if close_1d[i] <= teeth_aligned[i] or lips_aligned[i] <= jaw_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Short: exit when price >= Teeth or Lips >= Jaw
                if close_1d[i] >= teeth_aligned[i] or lips_aligned[i] >= jaw_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals