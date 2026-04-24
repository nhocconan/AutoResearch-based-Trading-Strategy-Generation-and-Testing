#!/usr/bin/env python3
"""
Hypothesis: Daily 10-period RSI mean reversion with 1-week ADX regime filter and volume spike confirmation.
- Primary timeframe: 1d for execution, HTF: 1w for ADX trend strength.
- ADX > 25 indicates trending market (avoid mean reversion in strong trends), ADX < 20 indicates ranging (favor mean reversion).
- Entry: Long when RSI(10) < 30 AND volume spike AND ADX < 20 (oversold bounce in ranging market).
         Short when RSI(10) > 70 AND volume spike AND ADX < 20 (overbought reversal in ranging market).
         In trending markets (ADX >= 25): avoid mean reversion entries to prevent whipsaw.
- Exit: Opposite RSI extreme (RSI > 70 for long exit, RSI < 30 for short exit) OR ADX shifts to trending (>=25).
- Volume confirmation: current volume > 1.5 * 20-period volume MA (to filter low-volume false signals).
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe.
- Works in both bull and bear: ranging markets occur in all regimes, and volume spikes often precede reversals.
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
    
    # RSI (10-period) on 1d
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/10, adjust=False, min_periods=10).mean().values
    avg_loss = loss.ewm(alpha=1/10, adjust=False, min_periods=10).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA (on 1d)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(30, 20)  # Need enough 1w bars for ADX and 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(adx_aligned[i]) or np.isnan(rsi[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        adx_val = adx_aligned[i]
        curr_rsi = rsi[i]
        
        if position == 0:
            # Check for entry signals - only in ranging market (ADX < 20)
            if volume_spike[i] and adx_val < 20:
                if curr_rsi < 30:  # Oversold: long
                    signals[i] = 0.25
                    position = 1
                elif curr_rsi > 70:  # Overbought: short
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: RSI > 70 (overbought) OR ADX shifts to trending (>=25)
            if curr_rsi > 70 or adx_val >= 25:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: RSI < 30 (oversold) OR ADX shifts to trending (>=25)
            if curr_rsi < 30 or adx_val >= 25:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_RSI10_1wADXRegime_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0