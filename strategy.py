#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with 1w ADX trend filter and 1d volume spike confirmation.
- Primary timeframe: 6h for execution, HTF: 1w for ADX trend filter (regime), 1d for volume confirmation.
- Entry: Price breaks above Donchian(20) upper band (long) or below lower band (short) on 6h close,
         with 1d volume > 2.0x 20-period volume MA AND 1w ADX(14) > 25 (strong trend).
- Exit: Price returns to Donchian(20) midpoint OR ADX drops below 20 (trend weakening).
- Uses discrete signal size 0.28 to balance return and drawdown.
- Target: 80-160 total trades over 4 years (20-40/year) for 6h timeframe.
- Works in bull via buying breakouts in strong uptrends, in bear via selling breakdowns in strong downtrends.
- ADX filter prevents whipsaws in ranging markets; volume confirmation reduces false breakouts.
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
    
    # Calculate 1w ADX(14) for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align length
    
    # Plus Directional Movement (+DM) and Minus Directional Movement (-DM)
    up_move = high_1w[1:] - high_1w[:-1]
    down_move = low_1w[:-1] - low_1w[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[0.0], plus_dm])
    minus_dm = np.concatenate([[0.0], minus_dm])
    
    # Smoothed TR, +DM, -DM using Wilder's smoothing (alpha = 1/period)
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value: simple average
        result[period-1] = np.nansum(data[:period])
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    atr_1w = wilder_smooth(tr, 14)
    plus_di_1w = 100 * wilder_smooth(plus_dm, 14) / atr_1w
    minus_di_1w = 100 * wilder_smooth(minus_dm, 14) / atr_1w
    dx_1w = 100 * np.abs(plus_di_1w - minus_di_1w) / (plus_di_1w + minus_di_1w)
    adx_1w = wilder_smooth(dx_1w, 14)
    
    # Align 1w ADX to 6h timeframe (completed 1w bar only)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # Calculate 1d volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    volume_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = volume_1d > (2.0 * volume_ma_1d)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    # Calculate 6h Donchian(20) channels
    donchian_window = 20
    upper_band = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    lower_band = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    midpoint = (upper_band + lower_band) / 2
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(donchian_window, 20) + 1  # Need Donchian(20), volume MA(20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(adx_1w_aligned[i]) or np.isnan(volume_spike_1d_aligned[i]) or 
            np.isnan(upper_band[i]) or np.isnan(lower_band[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above Donchian upper band with volume spike AND strong uptrend (ADX > 25)
            if (close[i] > upper_band[i] and volume_spike_1d_aligned[i] and 
                adx_1w_aligned[i] > 25):
                signals[i] = 0.28
                position = 1
            # Short: Price breaks below Donchian lower band with volume spike AND strong downtrend (ADX > 25)
            elif (close[i] < lower_band[i] and volume_spike_1d_aligned[i] and 
                  adx_1w_aligned[i] > 25):
                signals[i] = -0.28
                position = -1
        elif position == 1:
            # Long exit: Price returns to Donchian midpoint OR ADX drops below 20 (trend weakening)
            if (close[i] < midpoint[i] or adx_1w_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.28
        elif position == -1:
            # Short exit: Price returns to Donchian midpoint OR ADX drops below 20 (trend weakening)
            if (close[i] > midpoint[i] or adx_1w_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.28
    
    return signals

name = "6h_Donchian20_1wADX_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0