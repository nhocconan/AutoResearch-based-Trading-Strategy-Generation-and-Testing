#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Camarilla pivot breakout with 1d volume confirmation and ADX trend filter.
    # Long when price breaks above R4 AND 1d volume > 1.5x 20-period MA AND ADX > 25.
    # Short when price breaks below S4 AND 1d volume > 1.5x 20-period MA AND ADX > 25.
    # Exit when price retests the 1d pivot point (mean of high/low/close).
    # Uses discrete position sizing (0.25) to target 50-150 trades over 4 years.
    # Works in bull/bear via ADX filter ensuring we only trade strong trends, avoiding chop.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for pivot, volume and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d Camarilla pivot levels
    # Pivot point = (H + L + C) / 3
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    # Range = H - L
    range_1d = high_1d - low_1d
    # Resistance levels: R4 = C + ((H-L) * 1.1/2)
    r4_1d = close_1d + (range_1d * 1.1 / 2)
    # Support levels: S4 = C - ((H-L) * 1.1/2)
    s4_1d = close_1d - (range_1d * 1.1 / 2)
    
    # Calculate 1d volume 20-period MA
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d ADX (14-period)
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    up_move[0] = 0
    down_move[0] = 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di_14 = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr_14
    minus_di_14 = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr_14
    
    # ADX calculation
    dx = 100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14)
    dx = np.where(np.isnan(dx) | (plus_di_14 + minus_di_14 == 0), 0, dx)
    adx_14 = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align 1d indicators to 6h timeframe
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    adx_14_aligned = align_htf_to_ltf(prices, df_1d, adx_14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(pivot_1d_aligned[i]) or np.isnan(r4_1d_aligned[i]) or 
            np.isnan(s4_1d_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]) or 
            np.isnan(adx_14_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.5x 20-period average
        volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
        volume_spike = volume_1d_aligned[i] > 1.5 * vol_ma_1d_aligned[i]
        
        # Trend filter: ADX > 25 indicates strong trend
        trend_filter = adx_14_aligned[i] > 25
        
        # Camarilla breakout conditions
        breakout_long = close[i] > r4_1d_aligned[i-1]  # Break above previous period R4
        breakout_short = close[i] < s4_1d_aligned[i-1]  # Break below previous period S4
        
        # Exit conditions: price retests the 1d pivot point
        exit_long = close[i] < pivot_1d_aligned[i]
        exit_short = close[i] > pivot_1d_aligned[i]
        
        # Entry conditions
        if breakout_long and volume_spike and trend_filter and position != 1:
            position = 1
            signals[i] = position_size
        elif breakout_short and volume_spike and trend_filter and position != -1:
            position = -1
            signals[i] = -position_size
        # Exit conditions
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_1d_camarilla_breakout_volume_adx_v1"
timeframe = "6h"
leverage = 1.0