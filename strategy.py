#!/usr/bin/env python3
"""
4h_1d_Camarilla_Breakout_Volume
Hypothesis: Uses daily Camarilla pivot levels with volume confirmation and monthly volatility regime filter.
Long when price breaks above H4 with volume > 1.5x 20-period average and monthly volatility > 50th percentile.
Short when price breaks below L4 with volume confirmation and monthly volatility > 50th percentile.
Exit when price returns to Pivot point or volatility regime shifts.
Works in both bull and bear markets by trading institutional levels with volume confirmation.
Target: 20-50 trades/year on 4h (80-200 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily Camarilla levels: H4, L4, Pivot
    # H4 = Close + 1.5 * (High - Low)
    # L4 = Close - 1.5 * (High - Low)
    # Pivot = (High + Low + Close) / 3
    hl_range = high_1d - low_1d
    H4 = close_1d + 1.5 * hl_range
    L4 = close_1d - 1.5 * hl_range
    Pivot = (high_1d + low_1d + close_1d) / 3.0
    
    # Get monthly data for volatility regime filter
    df_1M = get_htf_data(prices, '1M')
    if len(df_1M) < 20:
        # Fallback to daily volatility if monthly not available
        returns = np.diff(np.log(close_1d))
        vol = pd.Series(returns).rolling(window=20, min_periods=20).std() * np.sqrt(252)
        vol_percentile = pd.Series(vol).rolling(window=50, min_periods=20).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
        ).values
        vol_regime = vol_percentile > 50.0
    else:
        # Use monthly volatility
        close_1M = df_1M['close'].values
        returns_1M = np.diff(np.log(close_1M))
        vol_1M = pd.Series(returns_1M).rolling(window=12, min_periods=12).std() * np.sqrt(12)
        vol_percentile_1M = pd.Series(vol_1M).rolling(window=24, min_periods=12).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
        ).values
        vol_regime = align_htf_to_ltf(prices, df_1M, vol_percentile_1M > 50.0)
    
    # Align Camarilla levels to 4h timeframe
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    Pivot_aligned = align_htf_to_ltf(prices, df_1d, Pivot)
    
    # Volume confirmation on 4h
    vol_ma_20_4h = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_expansion = volume > (vol_ma_20_4h * 1.5)
    
    # Session filter: 08:00-20:00 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% of capital
    
    for i in range(50, n):
        # Skip if not in session or data not ready
        if not session_mask[i] or \
           np.isnan(H4_aligned[i]) or \
           np.isnan(L4_aligned[i]) or \
           np.isnan(Pivot_aligned[i]) or \
           (hasattr(vol_regime, '__len__') and len(vol_regime) > i and np.isnan(vol_regime[i])) or \
           (not hasattr(vol_regime, '__len__') and np.isnan(vol_regime)):
            signals[i] = 0.0
            continue
        
        # Get volatility regime value
        if hasattr(vol_regime, '__len__'):
            vol_ok = vol_regime[i] if i < len(vol_regime) else False
        else:
            vol_ok = vol_regime
        
        # Entry conditions: volatility regime + volume expansion + Camarilla breakout
        if vol_ok and volume_expansion[i]:
            # Long: price breaks above H4
            if close[i] > H4_aligned[i]:
                if position != 1:
                    position = 1
                    signals[i] = position_size
                else:
                    signals[i] = position_size
            # Short: price breaks below L4
            elif close[i] < L4_aligned[i]:
                if position != -1:
                    position = -1
                    signals[i] = -position_size
                else:
                    signals[i] = -position_size
            # Exit to pivot when price returns
            elif position == 1 and close[i] <= Pivot_aligned[i]:
                position = 0
                signals[i] = 0.0
            elif position == -1 and close[i] >= Pivot_aligned[i]:
                position = 0
                signals[i] = 0.0
            # Hold position
            elif position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        else:
            # Exit positions when conditions not met
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_1d_Camarilla_Breakout_Volume"
timeframe = "4h"
leverage = 1.0