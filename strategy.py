#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout + 1d ADX regime filter + volume confirmation.
- Primary timeframe: 4h for execution, HTF: 1d for ADX regime filter.
- Donchian breakout: price > upper band (long) or < lower band (short) of 20-period Donchian channel.
- Regime filter: ADX(14) > 25 = trending (only trade breakouts in trend direction), ADX < 20 = ranging (mean revert at Donchian extremes).
- Volume confirmation: current volume > 1.5x 20-period volume MA to avoid low-volatility false signals.
- Discrete signal size: 0.25 to balance return and drawdown control.
- Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.
- Works in bull via buying upward breakouts in uptrend, in bear via selling downward breakouts in downtrend, and mean reverting in ranges.
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
    tr = np.concatenate([[np.nan], tr])  # First value NaN
    
    # Directional Movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[0.0], plus_dm])
    minus_dm = np.concatenate([[0.0], minus_dm])
    
    # Smoothed values (Wilder's smoothing = EMA with alpha=1/period)
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
    
    # Calculate 4h Donchian(20) channels
    donchian_window = 20
    upper_band = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    lower_band = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    # Calculate 4h volume confirmation: current volume > 1.5 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(donchian_window, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(adx_aligned[i]) or np.isnan(upper_band[i]) or 
            np.isnan(lower_band[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Regime: ADX > 25 = trending, ADX < 20 = ranging
            if adx_aligned[i] > 25:
                # Trending regime: trade breakouts in trend direction
                if close[i] > upper_band[i] and volume_spike[i]:
                    # Upward breakout: go long
                    signals[i] = 0.25
                    position = 1
                elif close[i] < lower_band[i] and volume_spike[i]:
                    # Downward breakout: go short
                    signals[i] = -0.25
                    position = -1
            elif adx_aligned[i] < 20:
                # Ranging regime: mean revert at Donchian extremes
                if close[i] < lower_band[i] and volume_spike[i]:
                    # Oversold: buy
                    signals[i] = 0.25
                    position = 1
                elif close[i] > upper_band[i] and volume_spike[i]:
                    # Overbought: sell
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price crosses below midpoint or trend reversal
            midpoint = (upper_band[i] + lower_band[i]) / 2
            if close[i] < midpoint:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above midpoint or trend reversal
            midpoint = (upper_band[i] + lower_band[i]) / 2
            if close[i] > midpoint:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_1dADX_Regime_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0