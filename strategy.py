#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R with 1d ADX trend filter and volume confirmation.
Long when Williams %R < -80 (oversold) AND 1d ADX > 25 (trending) AND volume > 1.5x average.
Short when Williams %R > -20 (overbought) AND 1d ADX > 25 (trending) AND volume > 1.5x average.
Exit when Williams %R crosses above -50 for long or below -50 for short, or ATR-based stoploss.
Uses discrete position sizing (0.25) to minimize fee churn. Targets 12-30 trades/year per symbol.
Williams %R identifies reversal points in trending markets, effective when combined with ADX trend filter.
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
    
    # Load 6h data for Williams %R calculation - ONCE before loop
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 30:
        return np.zeros(n)
    
    close_6h = df_6h['close'].values
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    volume_6h = df_6h['volume'].values
    
    # Calculate Williams %R(14) on 6h data
    highest_high = pd.Series(high_6h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_6h).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_6h) / (highest_high - lowest_low)
    # Handle division by zero
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate ATR(14) on 6h data for stoploss
    tr1 = np.maximum(high_6h - low_6h, np.abs(high_6h - np.roll(close_6h, 1)))
    tr2 = np.abs(low_6h - np.roll(close_6h, 1))
    tr = np.maximum(tr1, tr2)
    tr[0] = high_6h[0] - low_6h[0]  # first bar
    atr_6h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Load 1d data for ADX trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX(14) on 1d data
    # True Range
    tr1_1d = np.maximum(high_1d - low_1d, np.abs(high_1d - np.roll(close_1d, 1)))
    tr2_1d = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1_1d, tr2_1d)
    tr_1d[0] = high_1d[0] - low_1d[0]  # first bar
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed DM and TR
    plus_dm_smooth = pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values
    tr_smooth = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    dx = np.where((plus_di + minus_di) == 0, 0, dx)
    adx_1d = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align 1d ADX to 6h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Volume average (20-period) on 6h timeframe
    vol_ma = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_6h, vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(williams_r[i]) or np.isnan(atr_6h[i]) or 
            np.isnan(adx_1d_aligned[i]) or np.isnan(vol_ma_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            continue
        
        # Use 6h close for price comparison
        price_6h = close_6h[i]
        vol_ma_val = vol_ma_aligned[i]
        
        if position == 0:
            # Long: Williams %R < -80 (oversold) AND ADX > 25 (trending) AND volume confirmation
            if (williams_r[i] < -80 and adx_1d_aligned[i] > 25 and 
                volume_6h[i] > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
                entry_price = price_6h
            # Short: Williams %R > -20 (overbought) AND ADX > 25 (trending) AND volume confirmation
            elif (williams_r[i] > -20 and adx_1d_aligned[i] > 25 and 
                  volume_6h[i] > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
                entry_price = price_6h
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Williams %R crosses above -50 OR ATR stoploss
                if williams_r[i] > -50:
                    exit_signal = True
                elif price_6h < entry_price - 2.5 * atr_6h[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Williams %R crosses below -50 OR ATR stoploss
                if williams_r[i] < -50:
                    exit_signal = True
                elif price_6h > entry_price + 2.5 * atr_6h[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_WilliamsR_1dADX_Volume_ATRStop"
timeframe = "6h"
leverage = 1.0