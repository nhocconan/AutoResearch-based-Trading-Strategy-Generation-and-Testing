#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_VolumeSpike_RegimeFilter
Hypothesis: 4h Donchian(20) breakout with volume spike and ADX regime filter.
Long when price breaks above upper band with volume spike and ADX>25.
Short when price breaks below lower band with volume spike and ADX>25.
Exit on opposite band touch.
Uses discrete sizing (0.30) to balance performance and fees.
Target: 20-40 trades/year (~80-160 over 4 years) to avoid fee drag.
Works in trending markets via breakouts and avoids range-bound whipsaws via ADX filter.
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
    
    # Get 4h data for Donchian calculations (primary timeframe)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Donchian(20) channels for each 4h bar
    upper_20 = np.full(len(close_4h), np.nan)
    lower_20 = np.full(len(close_4h), np.nan)
    
    for i in range(20, len(close_4h)):
        upper_20[i] = np.max(high_4h[i-20:i])
        lower_20[i] = np.min(low_4h[i-20:i])
    
    # Align Donchian levels to original timeframe
    upper_20_aligned = align_htf_to_ltf(prices, df_4h, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_4h, lower_20)
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_ma_20)
    
    # ADX regime filter: only trade when ADX > 25 (trending market)
    # Calculate ADX(14) on 4h data
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = max(0, high[i] - high[i-1])
            minus_dm[i] = max(0, low[i-1] - low[i])
            if plus_dm[i] < minus_dm[i]:
                plus_dm[i] = 0
            if minus_dm[i] < plus_dm[i]:
                minus_dm[i] = 0
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Wilder's smoothing
        atr = np.zeros_like(tr)
        atr[period] = np.nanmean(tr[1:period+1])
        for i in range(period+1, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1/period, adjust=False).mean().values / atr)
        minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1/period, adjust=False).mean().values / atr)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = pd.Series(dx).ewm(alpha=1/period, adjust=False).mean().values
        
        # Set first period values to NaN
        adx[:period] = np.nan
        return adx
    
    adx_4h = calculate_adx(high_4h, low_4h, close_4h, 14)
    adx_4h_aligned = align_htf_to_ltf(prices, df_4h, adx_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for calculations
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(upper_20_aligned[i]) or np.isnan(lower_20_aligned[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(adx_4h_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.30 if position == 1 else -0.30)
            continue
        
        if position == 0:
            # Long: price breaks above upper band with volume spike and ADX>25
            long_signal = (close[i] > upper_20_aligned[i]) and vol_spike[i] and (adx_4h_aligned[i] > 25)
            # Short: price breaks below lower band with volume spike and ADX>25
            short_signal = (close[i] < lower_20_aligned[i]) and vol_spike[i] and (adx_4h_aligned[i] > 25)
            
            if long_signal:
                signals[i] = 0.30
                position = 1
            elif short_signal:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.30
            # Exit: price touches lower band
            if close[i] < lower_20_aligned[i]:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.30
            # Exit: price touches upper band
            if close[i] > upper_20_aligned[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Donchian20_Breakout_VolumeSpike_RegimeFilter"
timeframe = "4h"
leverage = 1.0