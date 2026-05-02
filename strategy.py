#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) + 1d ADX regime filter + volume confirmation
# Elder Ray measures bull/bear power relative to EMA13: Bull Power = High - EMA13, Bear Power = Low - EMA13
# 1d ADX > 25 filters for trending markets (avoid choppy conditions)
# Volume confirmation (1.5x 20-period average) ensures strong participation
# Discrete position sizing (0.25) balances profit potential with fee drag minimization
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe
# Works in both bull and bear markets by only trading strong trends with volume confirmation

name = "6h_ElderRay_1dADX25_Trend_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 1d ADX(14)
    high_1d = pd.Series(df_1d['high'])
    low_1d = pd.Series(df_1d['low'])
    close_1d = pd.Series(df_1d['close'])
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = abs(high_1d - close_1d.shift(1))
    tr3 = abs(low_1d - close_1d.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr.rolling(window=14, min_periods=14).mean()
    
    # Directional Movement
    up_move = high_1d - high_1d.shift(1)
    down_move = low_1d.shift(1) - low_1d
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    plus_dm_series = pd.Series(plus_dm, index=high_1d.index)
    minus_dm_series = pd.Series(minus_dm, index=high_1d.index)
    
    # Smoothed DM and TR
    plus_di_1d = 100 * (plus_dm_series.ewm(alpha=1/14, adjust=False).mean() / atr_1d)
    minus_di_1d = 100 * (minus_dm_series.ewm(alpha=1/14, adjust=False).mean() / atr_1d)
    
    # DX and ADX
    dx_1d = 100 * abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    adx_1d = dx_1d.ewm(alpha=1/14, adjust=False).mean().values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate 6h EMA13 for Elder Ray
    close_s = pd.Series(close)
    ema13_6h = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema13_6h
    bear_power = low - ema13_6h
    
    # Volume confirmation (1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 40
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Only trade in trending markets (ADX > 25)
        is_trending = adx_1d_aligned[i] > 25
        
        if position == 0:  # Flat - look for new entries
            # Elder Ray long: Bull Power > 0 and increasing
            # Elder Ray short: Bear Power < 0 and decreasing
            bull_power_rising = bull_power[i] > bull_power[i-1]
            bear_power_falling = bear_power[i] < bear_power[i-1]
            
            if is_trending and bull_power[i] > 0 and bull_power_rising and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            elif is_trending and bear_power[i] < 0 and bear_power_falling and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Bull Power <= 0 or trend weakening (ADX < 20)
            if bull_power[i] <= 0 or adx_1d_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Bear Power >= 0 or trend weakening (ADX < 20)
            if bear_power[i] >= 0 or adx_1d_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals