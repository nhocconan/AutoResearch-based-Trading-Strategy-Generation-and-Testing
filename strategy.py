#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + volume confirmation + 1d ADX regime filter
# Long when: price breaks above Donchian(20) high AND volume > 1.5x 20-period average AND 1d ADX > 25 (trending)
# Short when: price breaks below Donchian(20) low AND volume > 1.5x 20-period average AND 1d ADX > 25 (trending)
# Uses Donchian for structure, volume for conviction, ADX to avoid whipsaws in ranging markets.
# Discrete sizing (0.25) minimizes fee churn. Works in bull/bear via trend filter.
# Timeframe: 4h (primary), HTF: 1d for ADX calculation.

name = "4h_Donchian20_Volume_1dADX_TrendFilter_v1"
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
    
    # Load HTF data ONCE before loop for 1d ADX calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough for ADX calculation
        return np.zeros(n)
    
    # Calculate 1d ADX (14-period)
    # True Range
    tr1 = pd.Series(df_1d['high']).diff().abs()
    tr2 = (pd.Series(df_1d['high']) - pd.Series(df_1d['close']).shift()).abs()
    tr3 = (pd.Series(df_1d['low']) - pd.Series(df_1d['close']).shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr.rolling(window=14, min_periods=14).mean()
    
    # Directional Movement
    up_move = pd.Series(df_1d['high']).diff()
    down_move = -pd.Series(df_1d['low']).diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed DM
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    
    # Directional Indicators
    plus_di_1d = 100 * plus_dm_smooth / atr_1d
    minus_di_1d = 100 * minus_dm_smooth / atr_1d
    
    # DX and ADX
    dx = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    adx_1d = dx.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    
    # Align 1d ADX to 4h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d.values)
    
    # Calculate Donchian(20) on 4h data
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # warmup for Donchian and volume MA
    
    for i in range(start_idx, n):
        # Skip if ADX not available
        if np.isnan(adx_1d_aligned[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_donch_high = donch_high[i]
        curr_donch_low = donch_low[i]
        curr_vol_ma = vol_ma[i]
        curr_adx = adx_1d_aligned[i]
        
        # Handle exits
        if position == 1:  # Long position
            # Exit conditions:
            # 1. Price falls below Donchian low
            # 2. ADX falls below 20 (trend weakening)
            if (curr_close < curr_donch_low or curr_adx < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions:
            # 1. Price rises above Donchian high
            # 2. ADX falls below 20 (trend weakening)
            if (curr_close > curr_donch_high or curr_adx < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Volume confirmation: volume > 1.5x 20-period average
            vol_confirm = curr_volume > 1.5 * curr_vol_ma
            
            # Long entry: price breaks above Donchian high AND volume confirmation AND ADX > 25 (trending)
            if (curr_close > curr_donch_high) and vol_confirm and (curr_adx > 25):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian low AND volume confirmation AND ADX > 25 (trending)
            elif (curr_close < curr_donch_low) and vol_confirm and (curr_adx > 25):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals