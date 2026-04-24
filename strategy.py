#!/usr/bin/env python3
"""
Hypothesis: 12h Williams %R Extreme with 1d ADX Trend Filter and Volume Spike.
- Primary timeframe: 12h for execution, HTF: 1d for ADX trend filter.
- Entry: Williams %R(14) crosses above -20 (long) or below -80 (short) on 12h close, with volume > 1.8x 20-period volume MA.
- Direction filter: only long when 1d ADX(14) > 25 (strong trend), only short when 1d ADX(14) > 25.
- Williams %R identifies overbought/oversold conditions; ADX ensures we trade only in strong trending markets.
- Volume confirmation reduces false breakouts.
- Exit: Williams %R returns to -50 (mean reversion) or ADX < 20 (trend weakens).
- Discrete signal size: 0.25 to balance return and drawdown control.
- Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.
- Works in bull via buying pullbacks in uptrend, in bear via selling rallies in downtrend.
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
    
    # Calculate 1d ADX(14) for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range (TR)
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period: no previous close
    
    # Calculate Directional Movement (+DM, -DM)
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm[0] = 0.0
    minus_dm[0] = 0.0
    
    # Smooth TR, +DM, -DM using Wilder's smoothing (EMA-like with alpha=1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value: simple average
        result[period-1] = np.mean(data[:period])
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr = wilders_smoothing(tr, 14)
    plus_di = 100 * wilders_smoothing(plus_dm, 14) / atr
    minus_di = 100 * wilders_smoothing(minus_dm, 14) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilders_smoothing(dx, 14)
    
    # Align ADX to 12h timeframe (completed 1d bar only)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate Williams %R(14) on 12h data
    def williams_r(high, low, close, period):
        highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
        wr = -100 * (highest_high - close) / (highest_high - lowest_low)
        return wr
    
    wr = williams_r(high, low, close, 14)
    
    # Volume confirmation: current volume > 1.8 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(14, 20) + 13  # Need Williams %R(14), volume MA(20), plus buffer for stability
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(wr[i]) or np.isnan(adx_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Williams %R crosses above -20 (from oversold) with volume spike AND strong trend (ADX > 25)
            if (wr[i] > -20 and wr[i-1] <= -20 and volume_spike[i] and adx_aligned[i] > 25):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -80 (from overbought) with volume spike AND strong trend (ADX > 25)
            elif (wr[i] < -80 and wr[i-1] >= -80 and volume_spike[i] and adx_aligned[i] > 25):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Williams %R returns to -50 (mean reversion) or trend weakens (ADX < 20)
            if (wr[i] < -50 or adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R returns to -50 (mean reversion) or trend weakens (ADX < 20)
            if (wr[i] > -50 or adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsR_Extreme_1dADX_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0