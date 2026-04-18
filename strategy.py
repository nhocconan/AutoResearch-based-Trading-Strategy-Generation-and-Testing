#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 12h ADX filter and volume confirmation.
# Donchian breakouts capture momentum in trending markets.
# 12h ADX > 25 ensures we trade only in strong trending markets.
# Volume spike (>1.5x 20-period average) confirms conviction.
# Works in bull markets (upward breakouts) and bear markets (downward breakouts).
# Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag.
name = "6h_Donchian20_12hADX_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 6h data for Donchian channels
    df_6h = get_htf_data(prices, '6h')
    
    # Calculate Donchian channels on 6h data
    high_6h = pd.Series(df_6h['high'].values)
    low_6h = pd.Series(df_6h['low'].values)
    
    donchian_high = high_6h.rolling(window=20, min_periods=20).max().values
    donchian_low = low_6h.rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to lower timeframe (6h)
    donchian_high_aligned = align_htf_to_ltf(prices, df_6h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_6h, donchian_low)
    
    # Get 12h data for ADX filter
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate ADX on 12h data
    high_12h = pd.Series(df_12h['high'].values)
    low_12h = pd.Series(df_12h['low'].values)
    close_12h = pd.Series(df_12h['close'].values)
    
    # True Range
    tr1 = high_12h - low_12h
    tr2 = abs(high_12h - close_12h.shift(1))
    tr3 = abs(low_12h - close_12h.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_12h = tr.rolling(window=14, min_periods=14).mean()
    
    # Directional Movement
    up_move = high_12h.diff()
    down_move = low_12h.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed DM
    plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean() / atr_12h)
    minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean() / atr_12h)
    
    # DX and ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx_12h = dx.ewm(alpha=1/14, adjust=False).mean().values
    
    # Align ADX to lower timeframe (6h)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # Calculate volume spike: current volume > 1.5 * 20-period average volume
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(adx_12h_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Breakout conditions
        bullish_breakout = close[i] > donchian_high_aligned[i]
        bearish_breakout = close[i] < donchian_low_aligned[i]
        
        # Strong trend filter: ADX > 25
        strong_trend = adx_12h_aligned[i] > 25
        
        if position == 0:
            # Long: Bullish breakout AND strong trend AND volume spike
            if bullish_breakout and strong_trend and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Bearish breakout AND strong trend AND volume spike
            elif bearish_breakout and strong_trend and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price retracement to midline OR ADX weakens
            midline = (donchian_high_aligned[i] + donchian_low_aligned[i]) / 2
            if close[i] < midline or adx_12h_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price retracement to midline OR ADX weakens
            midline = (donchian_high_aligned[i] + donchian_low_aligned[i]) / 2
            if close[i] > midline or adx_12h_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals