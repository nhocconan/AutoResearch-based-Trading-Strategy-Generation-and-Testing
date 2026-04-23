#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian channel breakout with volume confirmation and ADX regime filter.
- Uses Donchian(20) breakout for entry direction (long on upper band, short on lower band)
- Volume confirmation: current volume > 1.5x 20-period average to filter false breakouts
- ADX filter: only trade when ADX > 25 (trending market) to avoid choppy conditions
- Position size: 0.25 (discrete level to minimize fee churn)
- Target: 20-50 trades/year (80-200 over 4 years) to avoid fee drag
- Works in bull/bear via trend filter (ADX) and volume confirmation
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    upper_band = high_roll
    lower_band = low_roll
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * vol_ma
    
    # ADX calculation (14-period)
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed TR and DM
    tr_period = 14
    atr = pd.Series(tr).ewm(span=tr_period, adjust=False, min_periods=tr_period).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(span=tr_period, adjust=False, min_periods=tr_period).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(span=tr_period, adjust=False, min_periods=tr_period).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / (atr + 1e-10)
    di_minus = 100 * dm_minus_smooth / (atr + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).ewm(span=tr_period, adjust=False, min_periods=tr_period).mean().values
    
    # ADX filter: > 25 (trending market)
    adx_filter = adx > 25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 20, 14)  # Donchian, volume MA, ADX
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(upper_band[i]) or 
            np.isnan(lower_band[i]) or
            np.isnan(vol_ma[i]) or
            np.isnan(adx[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Donchian breakout signals
        breakout_up = close[i] > upper_band[i-1]  # Close above prior upper band
        breakout_down = close[i] < lower_band[i-1]  # Close below prior lower band
        
        if position == 0:
            # Long: Donchian upper breakout AND volume confirmation AND ADX > 25
            if breakout_up and volume_confirm[i] and adx_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: Donchian lower breakout AND volume confirmation AND ADX > 25
            elif breakout_down and volume_confirm[i] and adx_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Donchian lower band break OR ADX < 20 (trend weakening)
            if close[i] < lower_band[i] or adx[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Donchian upper band break OR ADX < 20 (trend weakening)
            if close[i] > upper_band[i] or adx[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_VolumeConfirm_ADXFilter"
timeframe = "4h"
leverage = 1.0