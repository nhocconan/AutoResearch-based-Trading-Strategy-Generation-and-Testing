#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla H3/L3 breakout with 1d EMA34 trend filter and volume confirmation.
- Primary timeframe: 6h for balanced trade frequency and noise reduction.
- HTF: 1d EMA34 for trend direction (bullish if close > EMA34, bearish if close < EMA34).
- Volume: Current 6h volume > 1.5 * 20-period 6h volume MA to capture institutional interest.
- Camarilla: Calculated from prior 1d OHLC; H3/L3 as breakout levels.
- Entry: Long when close breaks above H3 AND 1d EMA34 bullish AND volume spike.
         Short when close breaks below L3 AND 1d EMA34 bearish AND volume spike.
- Exit: Opposite breakout (close below L3 for long, above H3 for short) or loss of volume confirmation.
- Signal size: 0.25 discrete to limit drawdown and reduce fee churn.
- Target: 75-150 total trades over 4 years (19-37/year) for 6h timeframe.
This strategy captures institutional breakouts in the direction of the daily trend,
filtered by volume to avoid false breakouts. Works in both bull and bear markets by
only taking trades aligned with the 1d trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla calculation and EMA34 trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    df_1d_close = df_1d['close'].values
    ema_1d = pd.Series(df_1d_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate Camarilla levels from prior 1d OHLC
    # H3, L3, H4, L4 based on prior day's range
    df_1d_high = df_1d['high'].values
    df_1d_low = df_1d['low'].values
    df_1d_close_prev = df_1d_close  # same array for close
    
    # Camarilla calculations
    rang = df_1d_high - df_1d_low
    H3 = df_1d_close_prev + (rang * 1.1 / 4)
    L3 = df_1d_close_prev - (rang * 1.1 / 4)
    H4 = df_1d_close_prev + (rang * 1.1 / 2)
    L4 = df_1d_close_prev - (rang * 1.1 / 2)
    
    # Calculate 20-period 6h volume MA from 1d volume (scaled)
    # Approximate: 1d volume distributed across 4x 6h bars
    df_1d_volume = df_1d['volume'].values
    vol_ma_1d = pd.Series(df_1d_volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    # Scale for 6h: 1d volume / 4 per 6h bar (approximation)
    vol_ma_6h_approx = vol_ma_1d / 4.0
    
    # Align HTF indicators to 6h
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    vol_ma_6h_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_6h_approx)
    
    # Volume confirmation: current 6h volume > 1.5 * 20-period scaled volume MA
    volume_spike = volume > (1.5 * vol_ma_6h_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 34  # Need EMA34 and Camarilla (which uses same lookback)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        if position == 0:
            # Check for breakout signals with volume spike
            if volume_spike[i]:
                # Bullish breakout: close > H3 AND 1d EMA34 bullish (close > EMA)
                if curr_close > H3_aligned[i] and curr_close > ema_1d_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Bearish breakout: close < L3 AND 1d EMA34 bearish (close < EMA)
                elif curr_close < L3_aligned[i] and curr_close < ema_1d_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: close < L3 (breakdown) OR loss of volume confirmation
            if curr_close < L3_aligned[i] or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: close > H3 (breakout) OR loss of volume confirmation
            if curr_close > H3_aligned[i] or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_H3L3_Breakout_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0