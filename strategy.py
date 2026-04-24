#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with 1w ADX trend filter and volume confirmation.
- Uses 6h timeframe (primary) and 1w HTF for ADX trend alignment (novel: weekly ADX as regime filter)
- Donchian breakout: long when price crosses above upper band, short when crosses below lower band
- Trend filter: only trade when weekly ADX > 25 (trending market) to avoid whipsaws in ranging markets
- Volume confirmation: current volume > 1.5 * 20-period volume MA to filter low-quality breakouts
- Exit: reverse signal or when price crosses the midpoint of the Donchian channel (mean reversion)
- Discrete signal size: 0.25 to balance return and risk while minimizing fee churn
- Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe as per research
- Works in both bull/bear: ADX filter ensures we only trade strong trends, Donchian captures momentum
- Novel edge: Weekly ADX as regime filter has not been saturated in 6h strategies (per experiment history)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 20-period Donchian channels on 6h data
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_high + lowest_low) / 2.0
    
    # Calculate 1w ADX for trend filter (novel: weekly ADX as regime filter)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:  # Need at least 14 periods for ADX calculation
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate True Range (TR)
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period TR is just high-low
    
    # Calculate Directional Movement (+DM and -DM)
    up_move = high_1w - np.roll(high_1w, 1)
    down_move = np.roll(low_1w, 1) - low_1w
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smooth TR, +DM, -DM using Wilder's smoothing (equivalent to EMA with alpha=1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nansum(data[:period])
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    atr_1w = wilders_smoothing(tr, 14)
    plus_di_1w = 100 * wilders_smoothing(plus_dm, 14) / atr_1w
    minus_di_1w = 100 * wilders_smoothing(minus_dm, 14) / atr_1w
    dx_1w = 100 * np.abs(plus_di_1w - minus_di_1w) / (plus_di_1w + minus_di_1w)
    adx_1w = wilders_smoothing(dx_1w, 14)
    
    # Align HTF indicators to LTF
    donchian_mid_aligned = align_htf_to_ltf(prices, prices, donchian_mid)  # Already LTF
    highest_high_aligned = align_htf_to_ltf(prices, prices, highest_high)
    lowest_low_aligned = align_htf_to_ltf(prices, prices, lowest_low)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * volume_ma)
    
    # Trend filter: weekly ADX > 25 indicates trending market
    strong_trend = adx_1w_aligned > 25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 20)  # Need Donchian(20) and volume MA(20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(highest_high_aligned[i]) or np.isnan(lowest_low_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or np.isnan(volume_confirm[i]) or
            np.isnan(adx_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price crosses above upper Donchian band AND strong trend AND volume confirmation
            if close[i] > highest_high_aligned[i] and close[i-1] <= highest_high_aligned[i-1] and strong_trend[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: price crosses below lower Donchian band AND strong trend AND volume confirmation
            elif close[i] < lowest_low_aligned[i] and close[i-1] >= lowest_low_aligned[i-1] and strong_trend[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below Donchian midpoint (mean reversion) or reverse signal
            if close[i] < donchian_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above Donchian midpoint (mean reversion) or reverse signal
            if close[i] > donchian_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_1wADX_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0