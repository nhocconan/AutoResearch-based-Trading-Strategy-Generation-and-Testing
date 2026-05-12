#!/usr/bin/env python3
# 4h_KAMA_Direction_Volume_Trend_Filter
# Hypothesis: Use 4h Kaufman Adaptive Moving Average (KAMA) for trend direction with volume confirmation
# and 1-day ADX regime filter to avoid choppy markets. KAMA adapts to market noise, reducing whipsaws
# in sideways markets while capturing trends. The 1-day ADX filter ensures we only trade when the
# daily trend is strong enough, working in both bull and bear markets by aligning with higher timeframe momentum.
# Volume confirmation adds conviction to signals. Designed for ~20-40 trades/year per symbol.

name = "4h_KAMA_Direction_Volume_Trend_Filter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 1d data for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)

    # Calculate ADX on 1d for trend strength
    # True Range
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
    
    # Directional Movement
    up_move = df_1d['high'] - df_1d['high'].shift(1)
    down_move = df_1d['low'].shift(1) - df_1d['low']
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    tr_mavg = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean()
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean()
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean()
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / tr_mavg
    minus_di = 100 * minus_dm_smooth / tr_mavg
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean()
    
    adx_values = adx.values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_values)
    
    # 4h KAMA calculation
    # Efficiency Ratio
    change = abs(close - np.roll(close, 10))
    change[0:10] = 0  # Avoid index issues
    volatility = np.sum(np.abs(np.diff(close, axis=0)), axis=0) if len(close) > 1 else 0
    # Simplified volatility calculation for 10-period
    volatility = pd.Series(close).rolling(window=10, min_periods=10).apply(
        lambda x: np.sum(np.abs(np.diff(x))), raw=True
    ).values
    volatility[0:10] = 1e-10  # Avoid division by zero
    
    er = change / volatility
    er[0:10] = 0  # Not enough data
    
    # Smoothing constants
    sc = (er * (0.6645 - 0.0645) + 0.0645) ** 2
    
    # KAMA calculation
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    kama_aligned = kama  # Already on 4h timeframe
    
    # Volume confirmation: current volume > 1.3x average of last 4 periods
    vol_ma = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    volume_ok = volume > (1.3 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(kama_aligned[i]) or np.isnan(adx_1d_aligned[i]) or 
            np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Regime filter: only trade when ADX > 25 (trending market)
        strong_trend = adx_1d_aligned[i] > 25
        
        if position == 0:
            # LONG: Close above KAMA AND strong trend AND volume
            if close[i] > kama_aligned[i] and strong_trend and volume_ok[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Close below KAMA AND strong trend AND volume
            elif close[i] < kama_aligned[i] and strong_trend and volume_ok[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close falls back below KAMA OR trend weakens
            if close[i] < kama_aligned[i] or not strong_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close rises back above KAMA OR trend weakens
            if close[i] > kama_aligned[i] or not strong_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals