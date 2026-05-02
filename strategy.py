#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and ADX regime filter
# Donchian channel breakouts capture sustained moves in both bull and bear markets
# Volume confirmation (>1.5 x 20-period EMA) filters false breakouts
# ADX > 25 ensures we only trade in trending regimes, avoiding whipsaws in ranging markets
# Discrete position sizing (0.25) controls fee drag while maintaining meaningful exposure
# Target: 75-200 total trades over 4 years (19-50/year) for optimal risk-adjusted returns
# Works in bull markets by catching upward breakouts, works in bear by catching downward breakouts

name = "4h_Donchian20_Breakout_VolumeConfirmation_ADX_Trend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume confirmation (volume spike > 1.5 x 20-period EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_confirmation = volume > (1.5 * vol_ema_20)
    
    # Donchian channel (20-period)
    high_rolling_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_rolling_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_upper = high_rolling_max
    donchian_lower = low_rolling_min
    
    # ADX calculation (14-period) for regime filter
    # Calculate True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0  # First bar has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Calculate Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = 0
    down_move[0] = 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed TR, +DM, -DM (using Wilder's smoothing = EMA with alpha=1/period)
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Calculate +DI and -DI
    plus_di = 100 * (plus_dm_smooth / atr)
    minus_di = 100 * (minus_dm_smooth / atr)
    
    # Calculate DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    dx = np.where((plus_di + minus_di) == 0, 0, dx)  # Avoid division by zero
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # ADX > 25 indicates trending regime
    adx_trend = adx > 25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for calculations)
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(volume_confirmation[i]) or np.isnan(adx_trend[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Close breaks above Donchian upper with volume confirmation and ADX trend
            if close[i] > donchian_upper[i] and volume_confirmation[i] and adx_trend[i]:
                signals[i] = 0.25
                position = 1
            # Short: Close breaks below Donchian lower with volume confirmation and ADX trend
            elif close[i] < donchian_lower[i] and volume_confirmation[i] and adx_trend[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Close drops below Donchian lower OR ADX loses trend strength
            if close[i] < donchian_lower[i] or not adx_trend[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Close rises above Donchian upper OR ADX loses trend strength
            if close[i] > donchian_upper[i] or not adx_trend[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals