#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 12h ADX trend filter and volume confirmation
# Uses Donchian channel breakouts (20-period) on 4h for trend following
# Confirmed by 12h ADX > 25 (strong trend) and volume spike (>1.5x average)
# Long when price breaks above upper Donchian + 12h ADX > 25 + volume confirmation
# Short when price breaks below lower Donchian + 12h ADX > 25 + volume confirmation
# Designed for ~20-30 trades/year with strong trend-following edge
# Works in bull markets via upward breakouts and in bear markets via downward breakdowns

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Donchian Channel (20-period) on 4h
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    donchian_middle = (donchian_upper + donchian_lower) / 2
    
    # Calculate ADX (14-period) on 12h for trend strength filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range calculation
    tr1 = np.abs(high_12h[1:] - low_12h[:-1])
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # Directional Movement
    up_move = np.concatenate([[np.nan], high_12h[1:] - high_12h[:-1]])
    down_move = np.concatenate([[np.nan], low_12h[:-1] - low_12h[1:]])
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smooth TR, +DM, -DM
    tr_smooth = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean().values
    
    # DI+ and DI-
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    
    # Volume moving average for confirmation (20-period EMA)
    vol_ma = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25
    
    for i in range(20, n):
        # Get aligned 12h indicators
        adx_aligned = align_htf_to_ltf(prices, df_12h, adx)[i]
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or np.isnan(adx_aligned) or np.isnan(vol_ma[i]):
            continue
        
        # Volume confirmation (1.5x average)
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Long signal: price breaks above upper Donchian + ADX > 25 + volume
        if position == 0 and close[i] > donchian_upper[i] and adx_aligned > 25 and volume_confirm:
            position = 1
            signals[i] = position_size
        # Short signal: price breaks below lower Donchian + ADX > 25 + volume
        elif position == 0 and close[i] < donchian_lower[i] and adx_aligned > 25 and volume_confirm:
            position = -1
            signals[i] = -position_size
        # Exit: price crosses middle Donchian
        elif position != 0:
            if position == 1 and close[i] < donchian_middle[i]:
                position = 0
                signals[i] = 0.0
            elif position == -1 and close[i] > donchian_middle[i]:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "4h_Donchian_12hADX_Volume"
timeframe = "4h"
leverage = 1.0