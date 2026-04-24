#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray (Bull/Bear Power) + 12h ADX regime filter.
- Primary timeframe: 6h for balanced trade frequency and noise reduction.
- HTF: 12h ADX(14) to filter regimes: ADX > 25 = trending, ADX < 20 = ranging.
- Elder Ray: Bull Power = High - EMA13(close), Bear Power = Low - EMA13(close).
- Entry: Long when Bull Power > 0 AND Bear Power rising (improving) AND 12h ADX > 25 (strong trend).
         Short when Bear Power < 0 AND Bull Power falling (deteriorating) AND 12h ADX > 25.
- Exit: Opposite Elder Ray signal (e.g., Bull Power <= 0 for long exit) OR ADX drops below 20 (regime shift to ranging).
- Signal size: 0.25 discrete to control drawdown and fee churn.
- Target: 80-160 total trades over 4 years (20-40/year) for 6h timeframe.
This strategy combines trend strength (ADX) with price momentum relative to equilibrium (Elder Ray).
In trending regimes (ADX>25), Elder Ray captures acceleration/deceleration of bull/bear power.
Avoids ranging markets where Elder Ray whipsaws by requiring ADX>25 for entry and exiting when ADX<20.
Works in both bull and bear markets by only taking trades in the direction of the 12h trend,
filtered by ADX to avoid false signals in low-momentum environments.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 12h data for ADX and EMA13 (for Elder Ray)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:  # Need enough for ADX and EMA
        return np.zeros(n)
    
    # Calculate 12h EMA13 for Elder Ray
    df_12h_close = df_12h['close'].values
    ema_12h = pd.Series(df_12h_close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate 12h ADX components
    df_12h_high = df_12h['high'].values
    df_12h_low = df_12h['low'].values
    df_12h_close_adx = df_12h['close'].values
    
    # True Range
    tr1 = df_12h_high - df_12h_low
    tr2 = np.abs(df_12h_high - np.roll(df_12h_close_adx, 1))
    tr3 = np.abs(df_12h_low - np.roll(df_12h_close_adx, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    dm_plus = np.where((df_12h_high - np.roll(df_12h_high, 1)) > (np.roll(df_12h_low, 1) - df_12h_low),
                       np.maximum(df_12h_high - np.roll(df_12h_high, 1), 0), 0)
    dm_minus = np.where((np.roll(df_12h_low, 1) - df_12h_low) > (df_12h_high - np.roll(df_12h_high, 1)),
                        np.maximum(np.roll(df_12h_low, 1) - df_12h_low, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed TR, DM+, DM- (Wilder's smoothing = EMA with alpha=1/period)
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / atr
    di_minus = 100 * dm_minus_smooth / atr
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    dx = np.where((di_plus + di_minus) == 0, 0, dx)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Elder Ray components
    bull_power = df_12h_high - ema_12h
    bear_power = df_12h_low - ema_12h
    
    # Align HTF indicators to 6h
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    bull_power_aligned = align_htf_to_ltf(prices, df_12h, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_12h, bear_power)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 30  # Need enough bars for ADX and EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_12h_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        if position == 0:
            # Check for entry signals in trending regime (ADX > 25)
            if adx_aligned[i] > 25:
                # Bullish entry: Bull Power > 0 AND Bear Power improving (less negative/rising)
                if bull_power_aligned[i] > 0 and bear_power_aligned[i] > bear_power_aligned[i-1]:
                    signals[i] = 0.25
                    position = 1
                # Bearish entry: Bear Power < 0 AND Bull Power deteriorating (falling)
                elif bear_power_aligned[i] < 0 and bull_power_aligned[i] < bull_power_aligned[i-1]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: Bull Power <= 0 OR regime shift to ranging (ADX < 20)
            if bull_power_aligned[i] <= 0 or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bear Power >= 0 OR regime shift to ranging (ADX < 20)
            if bear_power_aligned[i] >= 0 or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_BullBearPower_12hADX_RegimeFilter_v1"
timeframe = "6h"
leverage = 1.0