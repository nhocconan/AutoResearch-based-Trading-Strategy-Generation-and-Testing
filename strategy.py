#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1w ADX trend filter and volume confirmation.
- Primary timeframe: 12h to target 50-150 total trades over 4 years (12-37/year).
- HTF: 1w ADX(14) for trend strength (trending if ADX > 25, ranging if ADX < 20).
- Donchian levels: Upper and lower bands from prior 12h candle (using prior close to avoid look-ahead).
- Entry: Long when price breaks above prior upper band AND ADX > 25 AND volume > 1.5 * volume MA(50).
         Short when price breaks below prior lower band AND ADX > 25 AND volume > 1.5 * volume MA(50).
- Exit: Close-based reversal - exit long when price crosses below prior lower band,
        exit short when price crosses above prior upper band.
- Signal size: 0.30 discrete to balance return and drawdown.
This strategy captures medium-term breakouts in trending markets with volume confirmation,
designed to work in both bull and bear markets by filtering for trending conditions only.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for ADX trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate 1w ADX(14) for trend strength filter
    df_1w_high = df_1w['high'].values
    df_1w_low = df_1w['low'].values
    df_1w_close = df_1w['close'].values
    
    # Calculate True Range
    tr1 = df_1w_high - df_1w_low
    tr2 = np.abs(df_1w_high - np.roll(df_1w_close, 1))
    tr3 = np.abs(df_1w_low - np.roll(df_1w_close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period TR is just high-low
    
    # Calculate Directional Movement
    dm_plus = np.where((df_1w_high - np.roll(df_1w_high, 1)) > (np.roll(df_1w_low, 1) - df_1w_low),
                       np.maximum(df_1w_high - np.roll(df_1w_high, 1), 0), 0)
    dm_minus = np.where((np.roll(df_1w_low, 1) - df_1w_low) > (df_1w_high - np.roll(df_1w_high, 1)),
                        np.maximum(np.roll(df_1w_low, 1) - df_1w_low, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smooth TR, DM+, DM- using Wilder's smoothing (EMA with alpha=1/period)
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Calculate DI+ and DI-
    di_plus = 100 * dm_plus_smooth / (atr + 1e-10)
    di_minus = 100 * dm_minus_smooth / (atr + 1e-10)
    
    # Calculate DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Calculate prior 12h Donchian bands (using prior close to avoid look-ahead)
    # We need to get 12h data for the bands, but we'll use the current timeframe prices
    # and calculate Donchian from the last 20 periods (excluding current)
    lookback = 20
    donchian_upper = np.roll(pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values, 1)
    donchian_lower = np.roll(pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values, 1)
    # First value will be NaN due to roll, we'll handle it in the loop
    
    # Calculate volume MA(50) for confirmation
    vol_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    
    # Align HTF indicators to 12h
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(lookback + 1, 50, 30)  # Need enough bars for calculations
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(adx_aligned[i]) or np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        if position == 0:
            # Check for entry signals with volume confirmation (1.5x threshold) and ADX > 25
            vol_confirmed = curr_volume > 1.5 * vol_ma[i]
            strong_trend = adx_aligned[i] > 25
            
            # Long: Price breaks above prior upper band AND strong trend AND volume confirmed
            if curr_close > donchian_upper[i] and strong_trend and vol_confirmed:
                signals[i] = 0.30
                position = 1
            # Short: Price breaks below prior lower band AND strong trend AND volume confirmed
            elif curr_close < donchian_lower[i] and strong_trend and vol_confirmed:
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long when price crosses below prior lower band
            if curr_close < donchian_lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short when price crosses above prior upper band
            if curr_close > donchian_upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

name = "12h_Donchian20_1wADX_Trend_VolumeConfirmation_v1"
timeframe = "12h"
leverage = 1.0