#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Elder Ray Index with 1-day ADX regime filter
# Elder Ray: Bull Power = High - EMA13, Bear Power = EMA13 - Low
# Long when Bull Power > 0 and Bear Power < 0 (bullish momentum) + 1-day ADX > 25 (trending)
# Short when Bear Power > 0 and Bull Power < 0 (bearish momentum) + 1-day ADX > 25
# Exit when Bull Power and Bear Power have same sign (momentum divergence)
# Position size: 0.25 (25% of capital)
# Target: 75-150 total trades over 4 years (19-38/year)

name = "6h_elder_ray_1d_adx_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 1-day data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate EMA13 for Elder Ray
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema_13
    bear_power = ema_13 - low
    
    # Calculate 1-day ADX (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = np.diff(high_1d, prepend=high_1d[0])
    down_move = np.diff(low_1d, prepend=low_1d[0]) * -1  # invert to positive
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    tr_14 = pd.Series(tr_1d).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    plus_dm_14 = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    minus_dm_14 = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_14 / (tr_14 + 1e-10)
    minus_di = 100 * minus_dm_14 / (tr_14 + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(13, n):
        # Skip if required data not available
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: momentum divergence (both powers positive or both negative)
            if bull_power[i] > 0 and bear_power[i] > 0:
                signals[i] = 0.0
                position = 0
            elif bull_power[i] < 0 and bear_power[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: momentum divergence
            if bull_power[i] > 0 and bear_power[i] > 0:
                signals[i] = 0.0
                position = 0
            elif bull_power[i] < 0 and bear_power[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Elder Ray signal with ADX trend filter
            # Bullish: Bull Power > 0 and Bear Power < 0
            bullish = bull_power[i] > 0 and bear_power[i] < 0
            # Bearish: Bear Power > 0 and Bull Power < 0
            bearish = bear_power[i] > 0 and bull_power[i] < 0
            # Trend filter: 1-day ADX > 25
            trend_filter = adx_aligned[i] > 25
            
            if bullish and trend_filter:
                signals[i] = 0.25
                position = 1
            elif bearish and trend_filter:
                signals[i] = -0.25
                position = -1
    
    return signals