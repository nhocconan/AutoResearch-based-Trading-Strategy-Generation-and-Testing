#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with 1d ADX regime filter and volume confirmation.
- Primary timeframe: 6h for execution, HTF: 1d for ADX regime and Donchian channels.
- Regime filter: ADX(14) > 25 indicates trending market (use Donchian breakout).
                 ADX(14) < 20 indicates ranging market (fade at Donchian bands).
- Volume confirmation: current volume > 1.5x 20-period volume MA.
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-38/year) for 6h timeframe.
- Works in bull via buying breakouts in uptrend, in bear via selling breakdowns in downtrend,
  and in range via mean reversion at channel extremes.
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
    
    # Get 1d data for ADX regime and Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX(14) on 1d
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])  # align with original index
        
        # Directional Movement
        up_move = high[1:] - high[:-1]
        down_move = low[:-1] - low[1:]
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        plus_dm = np.concatenate([[0.0], plus_dm])
        minus_dm = np.concatenate([[0.0], minus_dm])
        
        # Smoothed TR, +DM, -DM (Wilder's smoothing)
        def Wilder_smoothing(data, period):
            result = np.full_like(data, np.nan)
            if len(data) < period:
                return result
            # First value is simple average
            result[period-1] = np.nanmean(data[:period])
            # Subsequent values: smoothed = prev_smoothed - (prev_smoothed/period) + current
            for i in range(period, len(data)):
                result[i] = result[i-1] - (result[i-1]/period) + data[i]
            return result
        
        atr = Wilder_smoothing(tr, period)
        plus_dm_smooth = Wilder_smoothing(plus_dm, period)
        minus_dm_smooth = Wilder_smoothing(minus_dm, period)
        
        # Directional Indicators
        plus_di = 100 * plus_dm_smooth / atr
        minus_di = 100 * minus_dm_smooth / atr
        
        # DX and ADX
        dx = np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100
        adx = Wilder_smoothing(dx, period)
        
        return adx
    
    adx_14 = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_14_aligned = align_htf_to_ltf(prices, df_1d, adx_14)
    
    # Calculate Donchian(20) channels on 1d
    def donchian_channels(high, low, period=20):
        upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
        return upper, lower
    
    donchian_upper, donchian_lower = donchian_channels(high_1d, low_1d, 20)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower)
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(30, 20)  # ADX + Donchian + volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(adx_14_aligned[i]) or np.isnan(donchian_upper_aligned[i]) or
            np.isnan(donchian_lower_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Regime-based logic
            if adx_14_aligned[i] > 25:  # Trending market - breakout
                if close[i] > donchian_upper_aligned[i] and volume_spike[i]:
                    signals[i] = 0.25
                    position = 1
                elif close[i] < donchian_lower_aligned[i] and volume_spike[i]:
                    signals[i] = -0.25
                    position = -1
            elif adx_14_aligned[i] < 20:  # Ranging market - mean reversion
                if close[i] > donchian_upper_aligned[i] and volume_spike[i]:
                    signals[i] = -0.25  # Sell at upper band
                    position = -1
                elif close[i] < donchian_lower_aligned[i] and volume_spike[i]:
                    signals[i] = 0.25   # Buy at lower band
                    position = 1
        elif position == 1:
            # Long exit: price returns to lower band or opposite breakout
            if not np.isnan(donchian_lower_aligned[i]):
                if close[i] < donchian_lower_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
        elif position == -1:
            # Short exit: price returns to upper band or opposite breakdown
            if not np.isnan(donchian_upper_aligned[i]):
                if close[i] > donchian_upper_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_1dADX_Regime_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0