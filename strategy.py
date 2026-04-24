#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R Extreme + 1d ADX Trend + Volume Spike Confirmation
- Primary timeframe: 6h for lower trade frequency (target 50-150 trades over 4 years).
- HTF: 1d ADX(14) for trend strength (>25 = trending, <20 = ranging).
- Entry: Long when Williams %R(14) < -80 (oversold) AND 1d ADX > 25 AND volume spike.
         Short when Williams %R(14) > -20 (overbought) AND 1d ADX > 25 AND volume spike.
- Exit: Williams %R returns to -50 (mean reversion) OR loss of volume confirmation OR ADX < 20 (trend weakens).
- Volume: Current 6h volume > 1.8 * 20-period 6h volume MA to capture participation.
- Signal size: 0.25 discrete to minimize fee churn and control drawdown.
- Works in both bull and bear markets by only taking mean-reversion trades in strong trends (ADX>25),
  avoiding choppy markets (ADX<20) where mean reversion fails. Volume spikes confirm institutional
  participation at extreme levels, reducing false signals.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 6h data for Williams %R and volume MA
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    # Calculate 6h Williams %R(14): (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(df_6h['high']).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(df_6h['low']).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - df_6h['close'].values) / (highest_high - lowest_low)
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate 20-period 6h volume MA
    volume_ma_6h = pd.Series(df_6h['volume']).rolling(window=20, min_periods=20).mean().values
    
    # Get 1d data for ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough for ADX calculation
        return np.zeros(n)
    
    # Calculate 1d ADX(14)
    # ADX = DX smoothed, DX = |DI+ - DI-| / (DI+ + DI-) * 100
    # DI+ = Smoothed((High - Prev High) when > (Prev Low - Low)) / ATR * 100
    # DI- = Smoothed((Prev Low - Low) when > (High - Prev High)) / ATR * 100
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Directional movement
    up_move = df_1d['high'] - df_1d['high'].shift(1)
    down_move = df_1d['low'].shift(1) - df_1d['low']
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed DM and TR
    smoothed_plus_dm = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    smoothed_minus_dm = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    smoothed_atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # DI+ and DI-
    plus_di = 100 * smoothed_plus_dm / smoothed_atr
    minus_di = 100 * smoothed_minus_dm / smoothed_atr
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    dx = np.where((plus_di + minus_di) == 0, 0, dx)  # Avoid division by zero
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align HTF indicators to 6h
    williams_r_aligned = align_htf_to_ltf(prices, df_6h, williams_r)
    volume_ma_6h_aligned = align_htf_to_ltf(prices, df_6h, volume_ma_6h)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume confirmation: current 6h volume > 1.8 * 20-period 6h volume MA (aligned)
    volume_spike = volume > (1.8 * volume_ma_6h_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(30, 20)  # Need enough bars for ADX and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r_aligned[i]) or np.isnan(volume_spike[i]) or 
            np.isnan(adx_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        
        if position == 0:
            # Check for entry signals with volume spike and strong trend (ADX > 25)
            if volume_spike[i] and adx_aligned[i] > 25:
                # Long when oversold (Williams %R < -80)
                if williams_r_aligned[i] < -80:
                    signals[i] = 0.25
                    position = 1
                # Short when overbought (Williams %R > -20)
                elif williams_r_aligned[i] > -20:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: Williams %R returns to -50 (mean reversion) OR weak trend OR loss of volume
            if williams_r_aligned[i] >= -50 or adx_aligned[i] < 20 or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R returns to -50 (mean reversion) OR weak trend OR loss of volume
            if williams_r_aligned[i] <= -50 or adx_aligned[i] < 20 or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_Extreme_1dADX_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0