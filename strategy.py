#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout with 1d volume confirmation and ADX trend filter.
# Long when: Price breaks above Donchian(20) high AND 1d volume > 1.5x 20-day avg volume AND 1d ADX > 25
# Short when: Price breaks below Donchian(20) low AND 1d volume > 1.5x 20-day avg volume AND 1d ADX > 25
# Exit when price returns to Donchian midpoint (mean of 20-period high/low).
# Designed for 12h timeframe with low trade frequency (target: 20-40/year) to avoid fee drag.
# Uses 1d for volume and ADX confirmation to ensure institutional participation and trend strength.
# Works in bull markets via upward breakouts, in bear markets via downward breakouts.
# Volume filter avoids breakouts on low conviction, ADX filter ensures trend strength.
name = "12h_Donchian20_1dVolume_ADX"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 12h Donchian(20) - breakout levels
    lookback = 20
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(lookback, n):
        highest_high[i] = np.max(high[i-lookback:i])
        lowest_low[i] = np.min(low[i-lookback:i])
    
    # Donchian midpoint for exit
    donchian_mid = (highest_high + lowest_low) / 2.0
    
    # 1d volume data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    vol_1d = df_1d['volume'].values
    
    # 1d volume > 1.5x 20-day average volume
    vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_surge = vol_1d > (1.5 * vol_ma_20)
    
    # 1d ADX > 25 for trend strength
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed DM and TR
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    atr_smooth = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / (atr_smooth + 1e-10)
    minus_di = 100 * minus_dm_smooth / (atr_smooth + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    strong_trend = adx > 25
    
    # Align 1d indicators to 12h timeframe
    vol_surge_aligned = align_htf_to_ltf(prices, df_1d, vol_surge)
    strong_trend_aligned = align_htf_to_ltf(prices, df_1d, strong_trend)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(lookback, 30)  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_surge_aligned[i]) or np.isnan(strong_trend_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above Donchian high + volume surge + strong trend
            long_condition = (close[i] > highest_high[i]) and vol_surge_aligned[i] and strong_trend_aligned[i]
            # Short: Price breaks below Donchian low + volume surge + strong trend
            short_condition = (close[i] < lowest_low[i]) and vol_surge_aligned[i] and strong_trend_aligned[i]
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Price returns to Donchian midpoint
            if close[i] <= donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price returns to Donchian midpoint
            if close[i] >= donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals