#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with 1d ADX regime filter and volume confirmation.
- Primary timeframe: 6h for execution, HTF: 1d for ADX trend strength.
- Donchian channels: upper/lower 20-period high/low from prior 6h candles.
- Breakout: Close > upper band (long) or Close < lower band (short) with volume > 1.5x 20-period volume MA.
- Regime filter: Only trade breakouts when 1d ADX > 25 (trending market) to avoid whipsaws in ranging markets.
- Volume confirmation ensures breakouts have institutional participation.
- Works in bull via buying breakouts in strong uptrends, in bear via selling breakdowns in strong downtrends.
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
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
    
    # Get 1d data for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough for ADX calculation
        return np.zeros(n)
    
    # Calculate 1d ADX (14-period)
    # ADX requires +DI, -DI, and TR calculations
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period has no previous close
    
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothing (Wilder's smoothing = EMA with alpha=1/period)
    def WilderSmooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            # First value is simple average
            result[period-1] = np.mean(data[:period])
            # Subsequent values: Wilder smoothing
            for i in range(period, len(data)):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    period = 14
    atr_1d = WilderSmooth(tr, period)
    plus_di_1d = 100 * WilderSmooth(plus_dm, period) / atr_1d
    minus_di_1d = 100 * WilderSmooth(minus_dm, period) / atr_1d
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    adx_1d = WilderSmooth(dx_1d, period)
    
    # Get 6h data for Donchian channels
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    # Calculate Donchian channels (20-period) from 6h data
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    donch_upper = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    donch_lower = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    
    # Align indicators to primary timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    donch_upper_aligned = align_htf_to_ltf(prices, df_6h, donch_upper)
    donch_lower_aligned = align_htf_to_ltf(prices, df_6h, donch_lower)
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(30, 20)  # ADX + Donchian + volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(donch_upper_aligned[i]) or
            np.isnan(donch_lower_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Check for Donchian breakout with volume spike and ADX regime filter
            if volume_spike[i] and adx_1d_aligned[i] > 25:
                # Long breakout: close > upper band and ADX > 25 (strong uptrend)
                if close[i] > donch_upper_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short breakdown: close < lower band and ADX > 25 (strong downtrend)
                elif close[i] < donch_lower_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price re-enters Donchian channel or opposite signal
            if close[i] < donch_lower_aligned[i]:  # Exit when price falls below lower band
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price re-enters Donchian channel or opposite signal
            if close[i] > donch_upper_aligned[i]:  # Exit when price rises above upper band
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_1dADX_Regime_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0