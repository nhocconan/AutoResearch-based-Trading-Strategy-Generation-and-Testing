#!/usr/bin/env python3
"""
12h_1d_Camarilla_Pivot_Breakout_Volume
Hypothesis: Combines Camarilla pivot levels from 1d with breakout confirmation on 12h and volume confirmation.
In low volatility (1d ATR < 20th percentile), waits for 12h close outside Camarilla H3/L3 levels with volume > 1.5x 20-period average.
Trades both bull and bear markets by trading volatility expansion after contraction.
Target: 12-37 trades/year on 12h (50-150 total over 4 years).
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
    
    # Get daily data for Camarilla pivot levels and volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Camarilla pivot levels for previous day
    # Pivot = (H + L + C) / 3
    # H3 = C + (H - L) * 1.1/2
    # L3 = C - (H - L) * 1.1/2
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    H3_1d = close_1d + range_1d * 1.1 / 2.0
    L3_1d = close_1d - range_1d * 1.1 / 2.0
    
    # Calculate 1d ATR for volatility filter (20-period)
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
    tr2 = np.maximum(np.abs(low_1d[1:] - close_1d[:-1]), tr1)
    tr = np.concatenate([[np.inf], tr2])  # First value inf
    atr_1d = pd.Series(tr).rolling(window=20, min_periods=20).mean()
    
    # Calculate 20-period percentile of ATR for volatility filter (20th percentile)
    atr_series = pd.Series(atr_1d.values)
    atr_percentile = atr_series.rolling(window=50, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    
    # Low volatility condition: ATR < 20th percentile
    low_vol = atr_percentile < 20.0
    
    # Get 12h data for breakout direction
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    volume_12h = df_12h['volume'].values
    
    # 12h breakout conditions: close outside Camarilla H3/L3 with volume expansion
    vol_ma_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean()
    volume_expansion_12h = volume_12h > (vol_ma_20_12h * 1.5)
    breakout_up = (close_12h > H3_1d) & volume_expansion_12h
    breakout_down = (close_12h < L3_1d) & volume_expansion_12h
    
    # Align all signals to 12h timeframe (using 12h as base timeframe)
    low_vol_aligned = align_htf_to_ltf(prices, df_1d, low_vol)
    breakout_up_aligned = align_htf_to_ltf(prices, df_12h, breakout_up)
    breakout_down_aligned = align_htf_to_ltf(prices, df_12h, breakout_down)
    H3_1d_aligned = align_htf_to_ltf(prices, df_1d, H3_1d)
    L3_1d_aligned = align_htf_to_ltf(prices, df_1d, L3_1d)
    
    # Session filter: 08:00-20:00 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% of capital
    
    for i in range(50, n):
        # Skip if not in session or data not ready
        if not session_mask[i] or \
           np.isnan(low_vol_aligned[i]) or \
           np.isnan(breakout_up_aligned[i]) or \
           np.isnan(breakout_down_aligned[i]) or \
           np.isnan(H3_1d_aligned[i]) or \
           np.isnan(L3_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Entry conditions: low volatility and price breaks Camarilla level with volume
        if low_vol_aligned[i]:
            # Volume confirmation on 12h
            vol_ma_20_12h = pd.Series(volume[:i+1]).rolling(window=20, min_periods=20).mean().iloc[-1] if i >= 20 else 0
            volume_expansion_12h = volume[i] > (vol_ma_20_12h * 1.5) if i >= 20 else False
            
            # Long entry: price breaks above H3
            if breakout_up_aligned[i] and volume_expansion_12h:
                if position != 1:
                    position = 1
                    signals[i] = position_size
                else:
                    signals[i] = position_size
            # Short entry: price breaks below L3
            elif breakout_down_aligned[i] and volume_expansion_12h:
                if position != -1:
                    position = -1
                    signals[i] = -position_size
                else:
                    signals[i] = -position_size
            # Hold position
            elif position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        else:
            # High volatility - exit any position
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_1d_Camarilla_Pivot_Breakout_Volume"
timeframe = "12h"
leverage = 1.0