#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 12h HMA21 trend filter and volume spike confirmation.
- Primary timeframe: 4h for execution, HTF: 12h for HMA trend.
- Donchian channels calculated from 20-period high/low on 4h.
- Entry: Long when price breaks above upper Donchian with volume spike and close > 12h HMA21 (uptrend).
         Short when price breaks below lower Donchian with volume spike and close < 12h HMA21 (downtrend).
- Exit: When price returns to the opposite Donchian level (mean reversion) or ATR-based stoploss.
- Works in bull via buying breakouts in uptrend, in bear via selling breakdowns in downtrend.
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_hma(series, period):
    """Calculate Hull Moving Average"""
    if len(series) < period:
        return np.full_like(series, np.nan, dtype=np.float64)
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    # WMA of half period
    wma_half = pd.Series(series).ewm(span=half_period, adjust=False, min_periods=half_period).mean()
    # WMA of full period
    wma_full = pd.Series(series).ewm(span=period, adjust=False, min_periods=period).mean()
    # Raw HMA
    raw_hma = 2 * wma_half - wma_full
    # Final HMA
    hma = pd.Series(raw_hma).ewm(span=sqrt_period, adjust=False, min_periods=sqrt_period).mean()
    return hma.values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for HMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 21:
        return np.zeros(n)
    
    # Calculate 12h HMA21 for trend filter
    hma_21 = calculate_hma(df_12h['close'], 21)
    
    # Align 12h HMA to 4h
    hma_21_aligned = align_htf_to_ltf(prices, df_12h, hma_21)
    
    # Donchian channels (20-period) on 4h
    lookback = 20
    upper = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lower = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume confirmation: current volume > 2.0 * 20-period volume MA (on 4h)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    # ATR for stoploss (optional - using mean reversion exit instead)
    # atr_period = 14
    # tr1 = pd.Series(high).rolling(window=2).max() - pd.Series(low).rolling(window=2).min()
    # tr2 = abs(pd.Series(high) - pd.Series(close).shift(1))
    # tr3 = abs(pd.Series(low) - pd.Series(close).shift(1))
    # tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    # atr = tr.rolling(window=atr_period, min_periods=atr_period).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(lookback, 20)  # Need enough bars for Donchian and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(hma_21_aligned[i]) or np.isnan(upper[i]) or 
            np.isnan(lower[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Check for breakout signals with volume spike and trend filter
            if volume_spike[i]:
                # Bullish breakout: price > upper Donchian and close > HMA21
                if close[i] > upper[i] and close[i] > hma_21_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Bearish breakdown: price < lower Donchian and close < HMA21
                elif close[i] < lower[i] and close[i] < hma_21_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price returns to lower Donchian (mean reversion)
            if close[i] <= lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price returns to upper Donchian (mean reversion)
            if close[i] >= upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_12hHMA21_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0