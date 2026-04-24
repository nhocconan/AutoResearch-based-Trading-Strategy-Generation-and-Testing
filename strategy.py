#!/usr/bin/env python3
"""
Hypothesis: 4h Williams %R extreme + 1d ADX regime filter + volume confirmation.
- Primary timeframe: 4h for execution, HTF: 1d for ADX regime filter.
- Williams %R(14): Long when <= -80 (oversold), Short when >= -20 (overbought).
- Regime filter: ADX(14) > 25 = trending (fade extremes), ADX < 20 = ranging (trade extremes).
  In trending markets, extremes often precede continuations; in ranging markets, extremes precede reversals.
- Volume confirmation: current volume > 1.8x 20-period volume MA to ensure participation.
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.
- Works in bull via buying oversold in uptrend (continuation) or selling overbought in range (reversion),
  and in bear via selling overbought in downtrend (continuation) or buying oversold in range (reversion).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d ADX(14) for regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # Directional Movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[0.0], plus_dm])
    minus_dm = np.concatenate([[0.0], minus_dm])
    
    # Smoothed values (Wilder's smoothing)
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    plus_di_smooth = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    minus_di_smooth = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # DI and DX
    plus_di = 100 * plus_di_smooth / atr
    minus_di = 100 * minus_di_smooth / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align 1d ADX to 4h timeframe (completed 1d bar only)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Williams %R(14) on 4h
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close) / (highest_high_14 - lowest_low_14)
    
    # Volume confirmation: current volume > 1.8 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(14, 34, 20)  # Williams %R(14) + ADX buffer + volume MA(20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(adx_aligned[i]) or 
            np.isnan(williams_r[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Regime: ADX > 25 = trending, ADX < 20 = ranging
            if adx_aligned[i] > 25:
                # Trending regime: fade extremes (expect continuation)
                if williams_r[i] <= -80 and volume_spike[i]:
                    # Oversold: buy expecting continuation of uptrend
                    signals[i] = 0.25
                    position = 1
                elif williams_r[i] >= -20 and volume_spike[i]:
                    # Overbought: sell expecting continuation of downtrend
                    signals[i] = -0.25
                    position = -1
            elif adx_aligned[i] < 20:
                # Ranging regime: trade extremes (expect reversal)
                if williams_r[i] <= -80 and volume_spike[i]:
                    # Oversold: buy expecting mean reversion up
                    signals[i] = 0.25
                    position = 1
                elif williams_r[i] >= -20 and volume_spike[i]:
                    # Overbought: sell expecting mean reversion down
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: Williams %R returns above -50 (momentum weakening)
            if williams_r[i] > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R returns below -50 (momentum weakening)
            if williams_r[i] < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsR_Extreme_1dADX_Regime_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0