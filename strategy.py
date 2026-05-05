#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and ADX trend filter
# Long when price breaks above 4h Donchian upper band AND volume > 1.5x 20-period average AND ADX > 25
# Short when price breaks below 4h Donchian lower band AND volume > 1.5x 20-period average AND ADX > 25
# Exit when price crosses 4h Donchian middle band (mean reversion to median)
# Uses Donchian channels for clear structure, volume for conviction, ADX to avoid choppy markets
# Timeframe: 4h, HTF: none needed for core logic. Designed to capture trends while avoiding false breakouts in low volatility.

name = "4h_Donchian20_Breakout_Volume_ADXFilter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate volume spike filter on 4h
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    # Calculate 4h Donchian channels (20-period)
    if len(high) >= 20:
        donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
        donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
        donchian_middle = (donchian_upper + donchian_lower) / 2
    else:
        donchian_upper = np.full(n, np.nan)
        donchian_lower = np.full(n, np.nan)
        donchian_middle = np.full(n, np.nan)
    
    # Calculate 4h ADX (14-period) for trend filter
    if len(high) >= 14:
        # True Range
        tr1 = np.abs(high[1:] - low[1:])
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
        
        # Directional Movement
        up_move = np.concatenate([[np.nan], high[1:] - high[:-1]])
        down_move = np.concatenate([[np.nan], low[:-1] - low[1:]])
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smoothed TR and DM
        tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
        plus_dm_14 = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values
        minus_dm_14 = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values
        
        # Directional Indicators
        plus_di_14 = 100 * plus_dm_14 / tr_14
        minus_di_14 = 100 * minus_dm_14 / tr_14
        
        # DX and ADX
        dx = 100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14)
        adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
        adx_filter = adx > 25
    else:
        adx = np.full(n, np.nan)
        adx_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or 
            np.isnan(donchian_middle[i]) or 
            np.isnan(adx[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian upper AND volume spike AND ADX > 25
            if (close[i] > donchian_upper[i] and 
                volume_filter[i] and 
                adx_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian lower AND volume spike AND ADX > 25
            elif (close[i] < donchian_lower[i] and 
                  volume_filter[i] and 
                  adx_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below Donchian middle (mean reversion)
            if close[i] < donchian_middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above Donchian middle (mean reversion)
            if close[i] > donchian_middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals