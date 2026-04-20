#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + Volume spike + Choppiness regime filter
# - Long when price breaks above Donchian(20) high + volume > 2x average + CHOP(14) > 61.8 (ranging)
# - Short when price breaks below Donchian(20) low + volume > 2x average + CHOP(14) > 61.8
# - Exit when price crosses Donchian midpoint (mean reversion in ranging markets)
# - Uses 1d ADX to filter out trending markets (only trade when ADX < 25)
# - Designed for 4h timeframe with selective entries to avoid overtrading
# - Target: 20-50 trades per year per symbol (80-200 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for ADX filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX(14) on 1d timeframe
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    dm_plus_smooth = pd.Series(dm_plus).rolling(window=14, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).rolling(window=14, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / atr
    di_minus = 100 * dm_minus_smooth / atr
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align ADX to 4h timeframe
    adx_4h = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate Donchian(20) on 4h
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    volume_4h = prices['volume'].values
    
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Volume spike: volume > 2x 20-period average
    vol_ma = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_4h > (2 * vol_ma)
    
    # Choppiness Index(14) on 4h
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values  # TR already calculated
    sum_atr_14 = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high_4h).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low_4h).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(sum_atr_14 / (max_high - min_low)) / np.log10(14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if NaN in indicators
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or \
           np.isnan(adx_4h[i]) or np.isnan(chop[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Regime filter: only trade in ranging markets (ADX < 25 and CHOP > 61.8)
        ranging = (adx_4h[i] < 25) and (chop[i] > 61.8)
        
        if position == 0:
            # Long entry: Donchian breakout up + volume spike + ranging
            if close_4h[i] > donchian_high[i] and volume_spike[i] and ranging:
                signals[i] = 0.25
                position = 1
            # Short entry: Donchian breakout down + volume spike + ranging
            elif close_4h[i] < donchian_low[i] and volume_spike[i] and ranging:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses Donchian midpoint or breaks down
            if close_4h[i] < donchian_mid[i] or close_4h[i] < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses Donchian midpoint or breaks up
            if close_4h[i] > donchian_mid[i] or close_4h[i] > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_VolumeSpike_ChopFilter_ADX"
timeframe = "4h"
leverage = 1.0