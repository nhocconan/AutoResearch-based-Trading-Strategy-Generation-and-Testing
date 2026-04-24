#!/usr/bin/env python3
"""
Hypothesis: Daily Williams %R (14) with 1-week ADX regime filter and volume confirmation.
- Primary timeframe: 1d for execution, HTF: 1w for ADX trend strength.
- ADX > 25 indicates trending market (momentum strategy), ADX < 20 indicates ranging (mean reversion at Williams %R extremes).
- Entry: Long when Williams %R crosses above -80 from below AND ADX > 25 (bullish momentum in trend).
         Short when Williams %R crosses below -20 from above AND ADX > 25 (bearish momentum in trend).
         In ranging (ADX < 20): Long when Williams %R < -80 and reverses up (%R > previous %R).
                                Short when Williams %R > -20 and reverses down (%R < previous %R).
- Exit: Opposite Williams %R signal or ADX regime shift to ranging.
- Volume confirmation: current volume > 1.2 * 20-period volume MA (to avoid false signals).
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe.
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
    
    # Get 1w data for ADX
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate ADX (14-period) on 1w
    # True Range
    tr1 = pd.Series(df_1w['high']).diff().abs()
    tr2 = (pd.Series(df_1w['high']) - pd.Series(df_1w['low'].shift())).abs()
    tr3 = (pd.Series(df_1w['low']) - pd.Series(df_1w['close'].shift())).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Movement
    up_move = pd.Series(df_1w['high']).diff()
    down_move = -pd.Series(df_1w['low']).diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed DM
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / atr
    minus_di = 100 * minus_dm_smooth / atr
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align 1w ADX to 1d
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # Williams %R (14-period) on 1d
    lookback_willr = 14
    highest_high = pd.Series(high).rolling(window=lookback_willr, min_periods=lookback_willr).max().values
    lowest_low = pd.Series(low).rolling(window=lookback_willr, min_periods=lookback_willr).min().values
    willr = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    
    # Volume confirmation: current volume > 1.2 * 20-period volume MA (on 1d)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.2 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(30, lookback_willr, 20)  # Need enough 1w bars for ADX and lookback for Williams %R
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(adx_aligned[i]) or np.isnan(willr[i]) or np.isnan(volume_spike[i]) or
            i == 0):  # Need previous Williams %R for reversal detection
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        adx_val = adx_aligned[i]
        curr_willr = willr[i]
        prev_willr = willr[i-1]
        
        if position == 0:
            # Check for entry signals
            if volume_spike[i]:
                if adx_val > 25:  # Trending regime: momentum strategy
                    # Bullish momentum: Williams %R crosses above -80 from below
                    if prev_willr <= -80 and curr_willr > -80:
                        signals[i] = 0.25
                        position = 1
                    # Bearish momentum: Williams %R crosses below -20 from above
                    elif prev_willr >= -20 and curr_willr < -20:
                        signals[i] = -0.25
                        position = -1
                else:  # Ranging regime (ADX < 20): mean reversion at extremes
                    # Long when Williams %R is oversold (< -80) and reverses up
                    if curr_willr < -80 and curr_willr > prev_willr:
                        signals[i] = 0.25
                        position = 1
                    # Short when Williams %R is overbought (> -20) and reverses down
                    elif curr_willr > -20 and curr_willr < prev_willr:
                        signals[i] = -0.25
                        position = -1
        elif position == 1:
            # Long exit: Williams %R crosses below -50 or ADX drops to ranging
            if curr_willr < -50 or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R crosses above -50 or ADX drops to ranging
            if curr_willr > -50 or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WilliamsR14_1wADXRegime_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0