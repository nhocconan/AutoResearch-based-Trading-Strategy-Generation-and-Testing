#!/usr/bin/env python3
"""
6h_ADX_DMI_Crossover_1dFilter
Hypothesis: On 6h timeframe, DMI+ crossing above DMI- with ADX > 20 signals trend strength. 
Use 1d timeframe to filter for bull/bear regime: only go long when 1d close > 1d EMA50, 
only short when 1d close < 1d EMA50. This avoids counter-trend trades in strong regimes.
Trades only when trend is aligned across timeframes, reducing whipsaws. Targets 15-35 trades/year.
Works in bull/bear markets by following the higher timeframe trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_dmi(high, low, close, period=14):
    """Calculate DMI+ and DMI-"""
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smooth using Wilder's smoothing
    tr_period = np.zeros_like(tr)
    dm_plus_period = np.zeros_like(dm_plus)
    dm_minus_period = np.zeros_like(dm_minus)
    
    tr_period[0] = tr[0]
    dm_plus_period[0] = dm_plus[0]
    dm_minus_period[0] = dm_minus[0]
    
    for i in range(1, len(tr)):
        tr_period[i] = tr_period[i-1] - (tr_period[i-1] / period) + tr[i]
        dm_plus_period[i] = dm_plus_period[i-1] - (dm_plus_period[i-1] / period) + dm_plus[i]
        dm_minus_period[i] = dm_minus_period[i-1] - (dm_minus_period[i-1] / period) + dm_minus[i]
    
    # Calculate DI+ and DI-
    di_plus = np.where(tr_period != 0, 100 * dm_plus_period / tr_period, 0)
    di_minus = np.where(tr_period != 0, 100 * dm_minus_period / tr_period, 0)
    
    return di_plus, di_minus

def calculate_adx(high, low, close, period=14):
    """Calculate Average Directional Index (ADX)"""
    di_plus, di_minus = calculate_dmi(high, low, close, period)
    
    # Calculate DX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    
    # Calculate ADX using Wilder's smoothing
    adx = np.zeros_like(dx)
    if len(dx) >= period:
        adx[period-1] = np.mean(dx[:period])  # First ADX value
        for i in range(period, len(dx)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
    
    return adx

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data once for EMA filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 6h DMI and ADX
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    adx_6h = calculate_adx(high, low, close, 14)
    di_plus_6h, di_minus_6h = calculate_dmi(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if np.isnan(adx_6h[i]) or np.isnan(ema_50_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # 1d trend filter: long only in uptrend, short only in downtrend
        bull_regime = close[i] > ema_50_1d_aligned[i]  # Using current close vs 1d EMA
        bear_regime = close[i] < ema_50_1d_aligned[i]
        
        # 6h signals: DMI crossover with ADX > 20
        bullish_cross = di_plus_6h[i] > di_minus_6h[i] and di_plus_6h[i-1] <= di_minus_6h[i-1]
        bearish_cross = di_minus_6h[i] > di_plus_6h[i] and di_minus_6h[i-1] <= di_plus_6h[i-1]
        strong_trend = adx_6h[i] > 20
        
        if position == 0:
            # Long: bullish crossover + strong trend + bull regime
            if bullish_cross and strong_trend and bull_regime:
                signals[i] = 0.25
                position = 1
            # Short: bearish crossover + strong trend + bear regime
            elif bearish_cross and strong_trend and bear_regime:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: bearish crossover or trend weakening
            if bearish_cross or adx_6h[i] < 15:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: bullish crossover or trend weakening
            if bullish_cross or adx_6h[i] < 15:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ADX_DMI_Crossover_1dFilter"
timeframe = "6h"
leverage = 1.0