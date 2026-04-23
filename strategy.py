#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1w ADX trend filter and volume confirmation.
Long when price breaks above Donchian upper band AND 1w ADX > 25 AND volume > 1.5x 20-period MA.
Short when price breaks below Donchian lower band AND 1w ADX > 25 AND volume > 1.5x 20-period MA.
Exit when price touches opposite Donchian band or 1w ADX < 20 (trend weakening).
Uses 1w HTF for trend strength filter to avoid weak/choppy markets, volume spike for momentum confirmation.
Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.
Donchian provides clear structure, 1w ADX filters regime, volume confirms breakout strength.
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
    
    # Calculate 12h Donchian channels (20-period)
    donchian_upper = np.full(n, np.nan)
    donchian_lower = np.full(n, np.nan)
    
    for i in range(20, n):
        # Use lookback of 20 periods (excluding current bar to avoid look-ahead)
        donchian_upper[i] = np.max(high[i-20:i])
        donchian_lower[i] = np.min(low[i-20:i])
    
    # Calculate 1w ADX for trend strength filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = np.abs(high_1w[1:] - low_1w[1:])
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index 0
    
    # Directional Movement
    up_move = high_1w[1:] - high_1w[:-1]
    down_move = low_1w[:-1] - low_1w[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smoothed TR, +DM, -DM (Wilder's smoothing = EMA with alpha=1/period)
    atr_period = 14
    atr = pd.Series(tr).ewm(alpha=1/atr_period, adjust=False, min_periods=atr_period).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/atr_period, adjust=False, min_periods=atr_period).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/atr_period, adjust=False, min_periods=atr_period).mean().values / atr
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1/atr_period, adjust=False, min_periods=atr_period).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # Calculate 12h volume MA (20-period) for spike filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 30 + atr_period, 20)  # Donchian (20), ADX needs 30+14, volume MA (20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        upper = donchian_upper[i]
        lower = donchian_lower[i]
        adx_val = adx_aligned[i]
        vol_ma_val = vol_ma_20[i]
        
        # Volume filter: 12h volume > 1.5x 20-period MA (adaptive to volatility)
        vol_filter = volume[i] > 1.5 * vol_ma_val
        
        if position == 0:
            # Long: Break above Donchian upper AND ADX > 25 (trending) AND volume filter
            if price > upper and adx_val > 25 and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Break below Donchian lower AND ADX > 25 (trending) AND volume filter
            elif price < lower and adx_val > 25 and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: price touches Donchian lower (opposite) OR ADX < 20 (trend weakening)
                if price < lower or adx_val < 20:
                    exit_signal = True
            elif position == -1:
                # Short exit: price touches Donchian upper (opposite) OR ADX < 20 (trend weakening)
                if price > upper or adx_val < 20:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_Donchian20_Breakout_1wADX_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0