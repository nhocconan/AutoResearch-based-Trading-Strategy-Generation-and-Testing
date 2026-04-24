#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R extreme reversal with 1d ADX trend filter and volume confirmation.
- Primary timeframe: 6h for lower trade frequency and better signal quality.
- HTF: 1d ADX for trend strength (ADX > 25 = trending market).
- Williams %R: Extreme oversold (< -80) for long, extreme overbought (> -20) for short.
- Volume: Current 6h volume > 1.5 * 20-period 6h volume MA to confirm participation.
- Entry: Long when Williams %R crosses above -80 AND 1d ADX > 25 AND volume spike.
         Short when Williams %R crosses below -20 AND 1d ADX > 25 AND volume spike.
- Exit: Williams %R returns to -50 (mean reversion) or loss of volume confirmation.
- Signal size: 0.25 discrete to balance return and drawdown.
- Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
This strategy captures mean reversion in strong trends, avoiding choppy markets via ADX filter.
Williams %R extremes signal exhaustion, with volume confirming institutional participation.
Works in both bull and bear markets by only trading in the direction of the 1d trend (via ADX > 25).
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
    
    # Get 1d data for ADX calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough for ADX calculation
        return np.zeros(n)
    
    # Calculate 1d ADX (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0  # First value has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    up_move[0] = 0
    down_move[0] = 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smooth TR and DM (Wilder's smoothing = EMA with alpha=1/period)
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Calculate 6h Williams %R (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate 6h volume MA (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Align HTF indicators to 6h
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(30, 20, 14)  # Need enough bars for ADX, volume MA, and Williams %R
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(adx_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_williams_r = williams_r[i]
        prev_williams_r = williams_r[i-1] if i > 0 else -50
        
        if position == 0:
            # Check for reversal signals with volume spike and strong trend (ADX > 25)
            if volume_spike[i] and adx_aligned[i] > 25:
                # Bullish reversal: Williams %R crosses above -80 from below
                if prev_williams_r <= -80 and curr_williams_r > -80:
                    signals[i] = 0.25
                    position = 1
                # Bearish reversal: Williams %R crosses below -20 from above
                elif prev_williams_r >= -20 and curr_williams_r < -20:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: Williams %R returns to -50 (mean reversion) or loss of volume/ADX
            if curr_williams_r >= -50 or not volume_spike[i] or adx_aligned[i] < 25:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R returns to -50 (mean reversion) or loss of volume/ADX
            if curr_williams_r <= -50 or not volume_spike[i] or adx_aligned[i] < 25:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_Extreme_1dADX_Trend_VolumeConfirmation_v1"
timeframe = "6h"
leverage = 1.0