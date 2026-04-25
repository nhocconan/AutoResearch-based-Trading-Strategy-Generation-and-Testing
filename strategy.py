#!/usr/bin/env python3
"""
12h_Donchian20_Breakout_1dADXRegime_VolumeFilter
Hypothesis: 12h Donchian(20) breakout with 1d ADX regime filter and volume confirmation. 
In trending markets (ADX>25), trade breakouts in direction of trend; in ranging markets (ADX<20), 
fade at Donchian extremes. Uses discrete sizing (0.25) to minimize fee churn. Target: 12-37 trades/year.
Works in bull via trend-following breakouts, in bear via mean reversion at extremes when trend weakens.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Donchian calculations (primary timeframe)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Donchian channels for each 12h bar (based on previous 20 bars)
    upper_12h = np.full(len(close_12h), np.nan)
    lower_12h = np.full(len(close_12h), np.nan)
    
    for i in range(20, len(close_12h)):
        upper_12h[i] = np.max(high_12h[i-20:i])  # Highest high of previous 20 bars
        lower_12h[i] = np.min(low_12h[i-20:i])   # Lowest low of previous 20 bars
    
    # Align Donchian levels to original timeframe
    upper_12h_aligned = align_htf_to_ltf(prices, df_12h, upper_12h)
    lower_12h_aligned = align_htf_to_ltf(prices, df_12h, lower_12h)
    
    # Get 1d data for regime filter (ADX)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d ADX for regime detection
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = max(0, high[i] - high[i-1])
            minus_dm[i] = max(0, low[i-1] - low[i])
            if plus_dm[i] == minus_dm[i]:
                plus_dm[i] = 0
                minus_dm[i] = 0
            elif plus_dm[i] < minus_dm[i]:
                plus_dm[i] = 0
            else:
                minus_dm[i] = 0
            
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Smooth using Wilder's smoothing (alpha = 1/period)
        atr = np.zeros_like(tr)
        plus_dm_smooth = np.zeros_like(plus_dm)
        minus_dm_smooth = np.zeros_like(minus_dm)
        
        atr[period-1] = np.mean(tr[1:period+1])
        plus_dm_smooth[period-1] = np.mean(plus_dm[1:period+1])
        minus_dm_smooth[period-1] = np.mean(minus_dm[1:period+1])
        
        for i in range(period, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            plus_dm_smooth[i] = (plus_dm_smooth[i-1] * (period-1) + plus_dm[i]) / period
            minus_dm_smooth[i] = (minus_dm_smooth[i-1] * (period-1) + minus_dm[i]) / period
        
        plus_di = 100 * plus_dm_smooth / (atr + 1e-10)
        minus_di = 100 * minus_dm_smooth / (atr + 1e-10)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        
        adx = np.zeros_like(dx)
        adx[2*period-1] = np.mean(dx[period:2*period])
        for i in range(2*period, len(dx)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for calculations
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(upper_12h_aligned[i]) or np.isnan(lower_12h_aligned[i]) or 
            np.isnan(adx_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        adx = adx_1d_aligned[i]
        
        if position == 0:
            # Regime-based entry logic
            if adx > 25:  # Trending regime
                # Long: break above upper Donchian with volume spike
                long_signal = (close[i] > upper_12h_aligned[i]) and vol_spike[i]
                # Short: break below lower Donchian with volume spike
                short_signal = (close[i] < lower_12h_aligned[i]) and vol_spike[i]
            else:  # Ranging regime (ADX < 25)
                # Long: mean reversion from lower Donchian with volume spike
                long_signal = (close[i] < lower_12h_aligned[i]) and vol_spike[i]
                # Short: mean reversion from upper Donchian with volume spike
                short_signal = (close[i] > upper_12h_aligned[i]) and vol_spike[i]
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit conditions: opposite touch or trend weakening
            exit_signal = (close[i] < lower_12h_aligned[i]) or (adx < 20 and close[i] > upper_12h_aligned[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions: opposite touch or trend weakening
            exit_signal = (close[i] > upper_12h_aligned[i]) or (adx < 20 and close[i] < lower_12h_aligned[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Donchian20_Breakout_1dADXRegime_VolumeFilter"
timeframe = "12h"
leverage = 1.0