#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1w ADX trend filter and volume confirmation.
# Long when price breaks above 12h Donchian upper band AND 1w ADX > 25 (strong trend) AND 12h volume > 1.5x 20-period volume MA.
# Short when price breaks below 12h Donchian lower band AND 1w ADX > 25 (strong trend) AND 12h volume > 1.5x 20-period volume MA.
# Exit on retracement to midpoint of Donchian channel or ADX weakening (< 20).
# Uses session filter (08-20 UTC) to avoid low-liquidity periods. Position size 0.25.
# Designed for 12h timeframe to achieve 50-150 total trades over 4 years (12-37/year) with strict entry conditions.
# Donchian channels provide clear structure, 1w ADX filters for higher-timeframe trend strength, volume confirms participation.
# Works in both bull and bear markets by only trading breakouts in the direction of the 1w trend when volume confirms.

name = "12h_Donchian20_1wADX_VolumeSpike_Session"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1w data for ADX trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w ADX for trend strength
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Plus Directional Movement (+DM) and Minus Directional Movement (-DM)
    up_move = high_1w - np.roll(high_1w, 1)
    down_move = np.roll(low_1w, 1) - low_1w
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smooth TR, +DM, -DM using Wilder's smoothing (equivalent to EMA with alpha=1/period)
    atr_1w = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    plus_di_1w = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr_1w * 100
    minus_di_1w = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr_1w * 100
    
    # ADX calculation
    dx = np.abs(plus_di_1w - minus_di_1w) / (plus_di_1w + minus_di_1w) * 100
    dx = np.where((plus_di_1w + minus_di_1w) == 0, 0, dx)
    adx_1w = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align 1w ADX to 12h timeframe
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # Calculate 12h Donchian(20) channels
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_middle = (donchian_upper + donchian_lower) / 2
    
    # Calculate 12h volume 20-period MA for spike detection
    volume_ma_12h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN or outside session
        if (np.isnan(adx_1w_aligned[i]) or np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or np.isnan(donchian_middle[i]) or 
            np.isnan(volume_ma_12h[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        
        # Volume spike condition: current 12h volume > 1.5x 20-period volume MA
        volume_spike = volume[i] > (volume_ma_12h[i] * 1.5)
        
        # Donchian breakout conditions
        breakout_up = high_val > donchian_upper[i]   # Price breaks above upper band
        breakout_down = low_val < donchian_lower[i]  # Price breaks below lower band
        
        # 1w trend strength condition
        strong_trend = adx_1w_aligned[i] > 25
        weak_trend = adx_1w_aligned[i] < 20
        
        if position == 0:
            # Long: Donchian breakout up AND strong trend AND volume spike AND session
            if breakout_up and strong_trend and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakout down AND strong trend AND volume spike AND session
            elif breakout_down and strong_trend and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price retouches middle band OR trend weakens
            if close_val < donchian_middle[i] or weak_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price retouches middle band OR trend weakens
            if close_val > donchian_middle[i] or weak_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals