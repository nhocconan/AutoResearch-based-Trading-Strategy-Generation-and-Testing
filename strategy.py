#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout with 1w ADX trend filter and volume confirmation.
# Donchian(20) breakout captures momentum; 1w ADX > 25 ensures strong trend;
# Volume spike confirms conviction. Works in bull and bear markets via long/short symmetry.
# Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag.
name = "12h_Donchian20_1wADX_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Donchian channels
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate Donchian channels on 12h data
    high_12h = pd.Series(df_12h['high'].values)
    low_12h = pd.Series(df_12h['low'].values)
    donchian_high = high_12h.rolling(window=20, min_periods=20).max().values
    donchian_low = low_12h.rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to lower timeframe (12h)
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    
    # Get 1w data for ADX filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate ADX on 1w data
    high_1w = pd.Series(df_1w['high'].values)
    low_1w = pd.Series(df_1w['low'].values)
    close_1w = pd.Series(df_1w['close'].values)
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = abs(high_1w - close_1w.shift(1))
    tr3 = abs(low_1w - close_1w.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1w = tr.rolling(window=14, min_periods=14).mean()
    
    # Directional Movement
    up_move = high_1w.diff()
    down_move = low_1w.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed DM
    plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean() / atr_1w)
    minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean() / atr_1w)
    
    # DX and ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx_1w = dx.ewm(alpha=1/14, adjust=False).mean().values
    
    # Align ADX to lower timeframe (12h)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # Calculate volume spike: current volume > 2.0 * 20-period average volume
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(adx_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        upper = donchian_high_aligned[i]
        lower = donchian_low_aligned[i]
        
        # Strong trend filter: ADX > 25
        strong_trend = adx_1w_aligned[i] > 25
        
        if position == 0:
            # Long: Breakout above upper band AND strong trend AND volume spike
            if close_val > upper and strong_trend and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Breakdown below lower band AND strong trend AND volume spike
            elif close_val < lower and strong_trend and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price crosses below midpoint OR ADX weakens
            midpoint = (upper + lower) / 2
            if close_val < midpoint or adx_1w_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price crosses above midpoint OR ADX weakens
            midpoint = (upper + lower) / 2
            if close_val > midpoint or adx_1w_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals