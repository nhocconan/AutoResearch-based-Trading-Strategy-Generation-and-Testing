#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h ADX trend filter with 1d Donchian breakout
# - ADX(14) > 25 indicates trending market on 1d timeframe
# - Long when price breaks above Donchian(20) upper band on 12h
# - Short when price breaks below Donchian(20) lower band on 12h
# - Only take trades when ADX confirms trend (avoid whipsaws in ranging markets)
# - Designed for 12h timeframe with selective entries to avoid overtrading
# - Target: 12-37 trades per year per symbol (50-150 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data for ADX calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX components on 1d timeframe
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0  # First period has no previous close
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    tr_period = 14
    atr = np.zeros_like(tr)
    dm_plus_smooth = np.zeros_like(dm_plus)
    dm_minus_smooth = np.zeros_like(dm_minus)
    
    # Initial values
    atr[tr_period-1] = np.mean(tr[:tr_period])
    dm_plus_smooth[tr_period-1] = np.mean(dm_plus[:tr_period])
    dm_minus_smooth[tr_period-1] = np.mean(dm_minus[:tr_period])
    
    # Wilder's smoothing
    for i in range(tr_period, len(tr)):
        atr[i] = (atr[i-1] * (tr_period - 1) + tr[i]) / tr_period
        dm_plus_smooth[i] = (dm_plus_smooth[i-1] * (tr_period - 1) + dm_plus[i]) / tr_period
        dm_minus_smooth[i] = (dm_minus_smooth[i-1] * (tr_period - 1) + dm_minus[i]) / tr_period
    
    # DI values
    plus_di = 100 * dm_plus_smooth / atr
    minus_di = 100 * dm_minus_smooth / atr
    
    # DX and ADX
    dx = np.zeros_like(atr)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    dx[np.isnan(dx) | np.isinf(dx)] = 0
    
    adx = np.zeros_like(dx)
    adx[2*tr_period-1] = np.mean(dx[tr_period:2*tr_period])
    for i in range(2*tr_period, len(dx)):
        adx[i] = (adx[i-1] * (tr_period - 1) + dx[i]) / tr_period
    
    # Align ADX to 12h timeframe
    adx_12h = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate Donchian channels on 12h
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    close_12h = prices['close'].values
    
    donchian_period = 20
    upper_band = pd.Series(high_12h).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lower_band = pd.Series(low_12h).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if NaN in indicators
        if np.isnan(adx_12h[i]) or np.isnan(upper_band[i]) or np.isnan(lower_band[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # ADX trend filter: only trade when ADX > 25
        is_trending = adx_12h[i] > 25
        
        if position == 0:
            # Long entry: price breaks above upper band + trending market
            if close_12h[i] > upper_band[i] and is_trending:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below lower band + trending market
            elif close_12h[i] < lower_band[i] and is_trending:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price falls below lower band or ADX weakens
            if close_12h[i] < lower_band[i] or adx_12h[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises above upper band or ADX weakens
            if close_12h[i] > upper_band[i] or adx_12h[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_ADX_TrendFilter_DonchianBreakout"
timeframe = "12h"
leverage = 1.0