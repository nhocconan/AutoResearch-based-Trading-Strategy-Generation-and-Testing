#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray Index with 1d ADX regime filter and volume confirmation.
- Primary timeframe: 6h for balanced trade frequency and noise reduction.
- HTF: 1d ADX(14) for regime detection (trending if ADX > 25, ranging if ADX < 20).
- Elder Ray: Bull Power = High - EMA13(Close), Bear Power = Low - EMA13(Close).
- Volume: Current 6h volume > 1.5 * 20-period volume MA to confirm participation.
- Entry: Long when Bull Power > 0 AND ADX > 25 (strong uptrend) AND volume spike.
         Short when Bear Power < 0 AND ADX > 25 (strong downtrend) AND volume spike.
- In ranging markets (ADX < 20): mean revert at extremes - Long when Bull Power < -0.5*ATR,
         Short when Bear Power > 0.5*ATR.
- Exit: Opposite Elder Ray signal or loss of volume/ADX confirmation.
- Signal size: 0.25 discrete to limit drawdown and reduce fee churn.
- Target: 80-120 total trades over 4 years (20-30/year) for 6h timeframe.
This combines trend following in strong markets with mean reversion in ranging markets,
using Elder Ray to measure price relative to underlying trend and ADX to adapt regime.
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
    
    # Calculate 6h EMA13 for Elder Ray
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components
    bull_power = high - ema_13  # Bull Power = High - EMA(Close)
    bear_power = low - ema_13   # Bear Power = Low - EMA(Close)
    
    # Calculate 6h ATR(14) for volatility normalization in ranging markets
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period TR is just high-low
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 6h volume MA(20)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Volume confirmation: current 6h volume > 1.5 * 20-period volume MA
    volume_spike = volume > (1.5 * vol_ma)
    
    # Get 1d data for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ADX(14)
    df_1d_high = df_1d['high'].values
    df_1d_low = df_1d['low'].values
    df_1d_close = df_1d['close'].values
    
    # True Range
    tr1_1d = df_1d_high - df_1d_low
    tr2_1d = np.abs(df_1d_high - np.roll(df_1d_close, 1))
    tr3_1d = np.abs(df_1d_low - np.roll(df_1d_close, 1))
    tr_1d = np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))
    tr_1d[0] = tr1_1d[0]
    
    # Directional Movement
    dm_plus = np.where((df_1d_high - np.roll(df_1d_high, 1)) > (np.roll(df_1d_low, 1) - df_1d_low),
                       np.maximum(df_1d_high - np.roll(df_1d_high, 1), 0), 0)
    dm_minus = np.where((np.roll(df_1d_low, 1) - df_1d_low) > (df_1d_high - np.roll(df_1d_high, 1)),
                        np.maximum(np.roll(df_1d_low, 1) - df_1d_low, 0), 0)
    
    # Smoothed TR, DM+, DM- using Wilder's smoothing (alpha=1/14)
    tr_14 = pd.Series(tr_1d).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align HTF indicators to 6h
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Session filter: 08:00-20:00 UTC (avoid low liquidity)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 14, 13)  # Need enough bars for volume MA, ATR, EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready or outside session
        if (np.isnan(adx_aligned[i]) or np.isnan(volume_spike[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(atr[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_bull = bull_power[i]
        curr_bear = bear_power[i]
        curr_adx = adx_aligned[i]
        curr_atr = atr[i]
        
        if position == 0:
            # Check for entry signals
            if volume_spike[i]:
                if curr_adx > 25:  # Trending market
                    # Bullish: Bull Power > 0 (price above EMA13) in strong uptrend
                    if curr_bull > 0:
                        signals[i] = 0.25
                        position = 1
                    # Bearish: Bear Power < 0 (price below EMA13) in strong downtrend
                    elif curr_bear < 0:
                        signals[i] = -0.25
                        position = -1
                elif curr_adx < 20:  # Ranging market
                    # Mean reversion at extremes
                    # Long when Bull Power is significantly negative (oversold)
                    if curr_bull < -0.5 * curr_atr:
                        signals[i] = 0.25
                        position = 1
                    # Short when Bear Power is significantly positive (overbought)
                    elif curr_bear > 0.5 * curr_atr:
                        signals[i] = -0.25
                        position = -1
        elif position == 1:
            # Long exit: Bull Power turns negative OR loss of confirmation OR outside session
            if curr_bull <= 0 or not volume_spike[i] or not in_session[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bear Power turns positive OR loss of confirmation OR outside session
            if curr_bear >= 0 or not volume_spike[i] or not in_session[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_1dADX_Regime_VolumeSpike_Session_v1"
timeframe = "6h"
leverage = 1.0